"""One authoritative fixed-step simulation shared by editor and OBS renderers."""
from __future__ import annotations

import math
import time
from .collision import constrain_angle

LIMITS = {"headX": (-20, 20), "headY": (-15, 15), "headZ": (-12, 12),
          "gazeX": (-1, 1), "gazeY": (-1, 1), "eyeOpen": (0, 1), "mouthOpen": (0, 1), "smile": (0, 1),
          "armLeft": (-30, 30), "armRight": (-30, 30),
          "elbowLeft": (-90, 90), "elbowRight": (-90, 90),
          "wristLeft": (-60, 60), "wristRight": (-60, 60)}
DEFAULTS = {k: (1.0 if k == "eyeOpen" else 0.0) for k in LIMITS}


class Runtime:
    def __init__(self):
        self.project = None
        self.mode = "mixed"
        self.owners = {k: "human" if (k.startswith("head") or k.startswith("gaze")) else "ai" for k in LIMITS}
        self.inputs = {"human": {}, "ai": {}}
        self.values = dict(DEFAULTS)
        self.expression = None
        self.expression_strength = 0.0
        self.expression_target = 0.0
        self.transition = .25
        self.motion = "idle"
        self.playing = True
        self.elapsed = 0.0
        self.hair = {"hairLeft": [0., 0.], "hairRight": [0., 0.]}
        self.sequence = 0

    def limits(self):
        limits=dict(LIMITS);s=(self.project or {}).get('parentMotion',{})
        if s.get('enabled'):
            for key,setting,default in [('headX','yawLimit',20),('headY','pitchLimit',15)]:
                value=s.get(setting,default);limits[key]=(-value,value)
        return limits

    def command(self, message, now=None):
        now = time.monotonic() if now is None else now
        if not isinstance(message, dict):
            raise ValueError("Expected JSON object")
        kind = message.get("type")
        if kind == "discover":
            return {"parameters": self.limits(), "expressions": self.project.get("expressions", []) if self.project else [],
                    "motions": ["idle", "nod", "wave", "tilt"], "timeoutSeconds": 1.5}
        if kind == "state":
            return self.snapshot()
        if kind == "parameters":
            source = message.get("source", "ai")
            if source not in self.inputs:
                raise ValueError("Unknown input source")
            params = message.get("values")
            if not isinstance(params, dict) or not params:
                raise ValueError("Missing values")
            for key, value in params.items():
                if key not in LIMITS or type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Invalid parameter: {key}")
            for key, value in params.items():
                low, high = self.limits()[key]
                self.inputs[source][key] = (max(low, min(high, value)), now)
        elif kind == "mode":
            mode = message.get("mode")
            owners = message.get("owners", self.owners)
            if mode not in ("human", "ai", "mixed") or not isinstance(owners, dict):
                raise ValueError("Invalid mode")
            if any(k not in LIMITS or v not in self.inputs for k, v in owners.items()):
                raise ValueError("Invalid ownership")
            self.mode = mode
            self.owners.update(owners)
        elif kind == "expression":
            expression = next((e for e in (self.project or {}).get("expressions", []) if e["id"] == message.get("expressionId")), None)
            strength = message.get("strength", 1)
            duration = message.get("duration", .25)
            if not expression or expression.get("stale"):
                raise ValueError("Unknown or stale expression")
            if type(strength) not in (int, float) or not 0 <= strength <= 1 or type(duration) not in (int, float) or not .01 <= duration <= 10:
                raise ValueError("Invalid strength or duration")
            self.expression, self.expression_target, self.transition = expression["id"], strength, duration
        elif kind == "motion":
            if message.get("name") not in ("idle", "nod", "wave", "tilt"):
                raise ValueError("Unsupported motion")
            self.motion = message["name"]
            self.playing = True
        elif kind == "pause":
            self.playing = not self.playing
        elif kind in ("release", "stop"):
            self.inputs["ai"].clear()
            if kind == "stop":
                self.inputs["human"].clear()
                self.motion = "idle"
                self.expression_target = 0
        else:
            raise ValueError("Unsupported command")
        return {"accepted": True}

    def step(self, dt=1/60, now=None):
        now = time.monotonic() if now is None else now
        self.sequence += 1
        if not self.playing:
            return
        self.elapsed += dt
        t = self.elapsed
        target = dict(DEFAULTS)
        target["headY"] = math.sin(t * .8) * 1.2
        blink = t % 4.8
        target["eyeOpen"] = min(1, abs(blink - .12) / .10) if blink < .24 else 1
        if self.motion == "nod":
            target["headY"] = math.sin(t * 3) * 8
        if self.motion == "wave":
            target["armRight"] = -18 + math.sin(t * 5) * 10
        if self.motion == "tilt":
            target["headZ"] = math.sin(t * 1.5) * 9
        for key in LIMITS:
            owner = self.owners[key] if self.mode == "mixed" else ("human" if self.mode == "human" else "ai")
            value, timestamp = self.inputs[owner].get(key, (target[key], -999))
            if now - timestamp < 1.5:
                target[key] = value
            low,high=self.limits()[key];target[key]=max(low,min(high,target[key]))
            self.values[key] += (target[key] - self.values[key]) * (1 - math.exp(-dt * 14))
        delta = self.expression_target - self.expression_strength
        self.expression_strength += max(-dt/self.transition, min(dt/self.transition, delta))
        moving=[p for p in (self.project or {}).get("parts",[]) if p.get("visible",True) and p["physics"]["enabled"]]
        for p in moving:self.hair.setdefault(p["id"],[0.,0.])
        for role, state in self.hair.items():
            part = next((p for p in moving if p["id"] == role), None)
            physics = (part or {}).get("physics", {})
            strength = physics.get("strength", .7) if physics.get("enabled", True) else 0
            force = (math.sin(t * 2 + (1 if role == "hairRight" else 0)) + self.values["headZ"] * .2) * strength
            state[1] += (force - physics.get("stiffness",36)*state[0] - physics.get("damping",9)*state[1]) * dt
            state[0] = max(-.06, min(.06, state[0] + state[1]*dt))
            rig=(self.project or {}).get('rig',{})
            if part and rig and not rig.get('stale'):
                tip=[sum(v[0] for v in part['polygon'])/len(part['polygon']),max(v[1] for v in part['polygon'])]
                angle,hit=constrain_angle(part['pivot'],tip,state[0],rig.get('colliders',[]))
                state[0]=angle
                if hit:state[1]=0

    def snapshot(self):
        return {"type": "frame", "sequence": self.sequence, "time": self.elapsed,
                "values": dict(self.values), "hair": {k: v[0] for k, v in self.hair.items()},
                "expressionId": self.expression, "strength": self.expression_strength,
                "mode": self.mode, "owners": dict(self.owners), "playing": self.playing,
                "projectId": self.project["id"] if self.project else None}
