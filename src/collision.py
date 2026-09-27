"""Angle constraint for a short hair strand against head/shoulder circle colliders."""
import math


def constrain_angle(pivot,tip,angle,colliders,limit=.3):
    dx,dy=tip[0]-pivot[0],tip[1]-pivot[1]
    def cost(a):
        c,s=math.cos(a),math.sin(a);penetration=0.
        for t in (.35,.65,1.):
            x=pivot[0]+t*(c*dx-s*dy);y=pivot[1]+t*(s*dx+c*dy)
            for collider in colliders:
                cx,cy=collider['center'];r=collider['radius']
                # Anchors embedded in the head may remain there; constrain only segments initially outside.
                rest=math.hypot(pivot[0]+dx*t-cx,pivot[1]+dy*t-cy)
                if rest<r:continue
                penetration+=max(0,r-math.hypot(x-cx,y-cy))**2
        return penetration
    if cost(angle)<1e-12:return angle,False
    candidates=[angle,0.]+[-limit+2*limit*i/120 for i in range(121)]
    feasible=[a for a in candidates if cost(a)<1e-12]
    resolved=min(feasible,key=lambda a:abs(a-angle)) if feasible else min(candidates,key=lambda a:cost(a))
    return resolved,True
