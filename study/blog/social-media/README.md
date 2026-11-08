# Social media sources

- `linkedin_post.txt`, `x_thread.txt`: the posted texts.
- `video_src/`: the LinkedIn/X video (`build.py` renders `scenes.html` layers with headless Chrome and composes the clips with ffmpeg; the clips come from `site/assets/video/`).
- `slides_src/`: the carousel slides (HTML rendered to 1080×1350 PNGs and a PDF).

Rendered outputs (MP4s, PNGs, PDF) are kept outside git.
