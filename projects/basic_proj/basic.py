import irsim

env = irsim.make("basic_world.yaml")
env.set_title("Multi-Robot Navigation Simulation")

for i in range(1000):
    env.step()  # Update simulation state
    env.render(0.05)  # Render with 0.05 second interval (20Hz)

    if env.done():  # Check if simulation should end
        break

env.end()  # Clean up resources