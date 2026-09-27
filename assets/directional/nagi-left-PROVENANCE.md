# Nagi shallow left view

Generated with built-in image_gen on 2026-09-26. References: the existing front face input and `nagi-right-2160.png`. New rendering, not a horizontal flip. The angle is an artistic approximation, not a calibrated camera measurement.

Original retained as `nagi-left-original.png`. FFmpeg Lanczos enlarged to `nagi-left-2160.png`, preserving RGBA. Upscaling does not reconstruct missing detail. This is a reference asset; registering correspondence points and inferring left-view depth are separate steps.

Prompt:

Create a SINGLE identity-preserving shallow LEFT-facing anime face reference for a 2D character rig. Image 1 is the authoritative front Nagi identity; image 2 is the existing RIGHT view to match style and approximate angle magnitude, but output must face the OPPOSITE direction: viewer's LEFT, nose slightly to the LEFT of canvas center. Turn head approximately 18 degrees LEFT, shallow oblique, not profile. Preserve the exact asymmetric teal bangs and hair part on the same anatomical sides as image 1: redraw the turned head, do not horizontally mirror image 2. Same teal wavy shoulder-length hair, golden eyes, delicate linework, soft shading, small closed-mouth gentle smile. Both open eyes visible, screen-left far eye subtly narrower, anime-designed cheeks/chin/nose. No pitch or head tilt. Match face scale/framing: square closeup, crown at top edge, chin ~81% down, neck to lower edge; no added clothes or accessories. Clean transparent background and alpha with no colored fringe. No text, grid, second face, watermark. Aim for at least 2160x2160 if supported.
