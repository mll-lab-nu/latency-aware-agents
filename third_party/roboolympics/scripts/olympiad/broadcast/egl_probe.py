"""Check actual hardware rendering, not just whether EGL can import."""
import mujoco
from OpenGL.GL import GL_RENDERER, glGetString

model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(mujoco.MjData(model))
    renderer.render()
    name = glGetString(GL_RENDERER).decode()
    if 'nvidia' not in name.lower():
        raise RuntimeError(f'Expected NVIDIA hardware rendering, got {name}')
    print(name)
