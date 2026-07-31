# Repository instructions

- Read `AGENTS.md` before modifying the repository.
- Work on the currently checked-out branch. Do not create or switch branches unless explicitly requested.
- Do not commit or push automatically.
- Do not modify NVIDIA, CUDA, JetPack, or native Jetson OpenCV packages.
- Do not add pip OpenCV packages to Jetson dependencies.
- Do not claim hardware validation unless the command ran on the Jetson.
- Keep camera access separate from calibration mathematics.
- Ordinary tests must run without a camera.
- Use `pathlib` and portable paths.
- Do not add dependencies without explaining why.
- Prefer small, readable standalone modules over large frameworks.
- Always report commands executed, checks passed, checks not performed,
  assumptions, and remaining risks.
