#!/usr/bin/env python3
"""Regenerate assets/nanomem-demo.gif from a real run of demo_stale.py.

    python3 assets/record_demo.py

Needs `agg` (brew install agg) for the GIF step; the .cast is written either
way. Nothing here fakes output or timing: the demo is run under a real pty and
every delay in the recording is a delay the program actually took. The only
thing the recorder arranges is the terminal size and NANOMEM_DEMO_PAUSE, which
inserts sleeps between blocks so the result is watchable. `--brief` drops the
explanatory prose; the computed lines are identical either way.
"""
import fcntl, json, os, pty, select, shutil, struct, subprocess, sys, termios, time

COLS, ROWS, PAUSE = 94, 35, "0.35"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CAST = os.path.join(HERE, "nanomem-demo.cast")
GIF = os.path.join(HERE, "nanomem-demo.gif")

SCRIPT = r"""
printf '\033[?25l'
sleep 0.6
printf '\033[32m$\033[0m python demo_stale.py --brief\n\n'
sleep 1.0
python demo_stale.py --brief
sleep 2.0
"""


def record():
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(REPO)
        os.environ.update(TERM="xterm-256color", NANOMEM_DEMO_PAUSE=PAUSE,
                          COLUMNS=str(COLS), LINES=str(ROWS))
        os.execvp("bash", ["bash", "-c", SCRIPT])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    events, start = [], time.time()
    while True:
        if not select.select([fd], [], [], 30)[0]:
            break
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        events.append([round(time.time() - start, 6), "o",
                       data.decode("utf-8", "replace")])
    os.close(fd)
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass

    with open(CAST, "w") as f:
        f.write(json.dumps({"version": 2, "width": COLS, "height": ROWS,
                            "timestamp": int(start),
                            "env": {"TERM": "xterm-256color",
                                    "SHELL": "/bin/bash"}}) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")
    return events[-1][0] if events else 0.0


def main():
    dur = record()
    print("recorded %.1fs -> %s" % (dur, CAST))
    if not shutil.which("agg"):
        print("agg not installed; skipping the GIF (brew install agg)")
        return 0
    subprocess.check_call(["agg", CAST, GIF, "--theme", "dracula",
                           "--font-size", "14", "--line-height", "1.4",
                           "--last-frame-duration", "4", "--fps-cap", "12",
                           "--font-family", "Menlo"])
    print("wrote %s (%.0f KB)" % (GIF, os.path.getsize(GIF) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
