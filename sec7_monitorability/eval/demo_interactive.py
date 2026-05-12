#!/usr/bin/env python3
"""
Interactive stream demo — type freely while the model thinks and responds in real time.

The model ticks at a fixed rate (default 1/s), consuming one user token per tick
and generating all 10 cognitive channels simultaneously.

Usage:
    python script/demo_interactive.py --model /path/to/checkpoint
    python script/demo_interactive.py --model /path/to/checkpoint --tick 0.5

Controls:
    Type + Space/Enter  — tokenize word and queue for model (Enter also unpauses)
    Esc                 — toggle pause / resume
    Ctrl+C              — quit
"""

import argparse
import collections
import curses
import os
import queue
import sys
import textwrap
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from stream_inference import (
    C,
    _tokenize_user,
    generate,
    load_model,
)

THINK_NAMES = ["Ana", "Ske", "Int", "Bet", "Cur", "Voi", "Ins", "Syn"]


# ── shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock = threading.Lock()
        # output
        self.output_history = collections.deque(
            maxlen=30
        )  # per-tick decoded token or "-"
        # thinking: per-channel history (deque of recent decoded tokens)
        self.think_history = [collections.deque(maxlen=30) for _ in range(8)]
        # user token tracking
        self.user_history = collections.deque(maxlen=30)  # per-tick: decoded str or "-"
        self.user_queued = []  # list of (token_id, decoded_str) waiting in queue
        # conversation log: list of ("user", text) or ("output", text)
        self.conv_log = []
        self._cur_speaker = None  # "user" or "output" or None
        self._cur_text = ""
        # status
        self.row = 0
        self.paused = True  # start paused
        self.done = False


# ── model thread ──────────────────────────────────────────────────────────────
class ModelRunner:
    def __init__(self, model, tokenizer, silence_token, state, token_queue, args):
        self.model = model
        self.tokenizer = tokenizer
        self.silence_token = silence_token
        self.state = state
        self.token_queue = token_queue
        self.tick = args.tick
        self.stop = threading.Event()
        self.gen_kwargs = dict(
            max_rows=10_000,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            silence_penalty=args.silence_penalty,
            think_silence_penalty=args.think_silence_penalty,
            skip_silence=args.skip_silence,
            warm_start=args.system_prompt,
            system_prompt_version=args.system_prompt_version,
        )

    def _decode(self, tok_id):
        if tok_id == self.silence_token:
            return "-"
        return self.tokenizer.decode([tok_id]).strip() or "-"

    def run(self):
        # Create the generator in interactive (send) mode — user_text=""
        gen = generate(
            self.model,
            self.tokenizer,
            "",  # empty = interactive mode
            self.silence_token,
            **self.gen_kwargs,
        )

        # Bootstrap: drain any prefill rows (warm_start header) so the
        # generator suspends at the first yield that captures .send() tokens.
        # Seed mode has 0 prefill rows; warm_start mode has n_prefill rows.
        while True:
            try:
                row_idx, row, is_prefill = next(gen)
            except StopIteration:
                return
            with self.state.lock:
                self.state.row = row_idx
                self.state.output_history.append(self._decode(row[1]))
                for i, c in enumerate(range(2, C)):
                    self.state.think_history[i].append(self._decode(row[c]))
                self.state.user_history.append(self._decode(row[0]))
            if not is_prefill:
                break  # suspended at the first non-prefill yield

        while not self.stop.is_set():
            # wait while paused
            while not self.stop.is_set():
                with self.state.lock:
                    if not self.state.paused:
                        break
                self.stop.wait(0.1)
            if self.stop.is_set():
                break

            t0 = time.monotonic()

            # pop one user token from the queue
            try:
                user_token, user_str = self.token_queue.get_nowait()
            except queue.Empty:
                user_token, user_str = self.silence_token, None

            # Advance the generator — send the user token
            try:
                row_idx, row, is_prefill = gen.send(user_token)
            except StopIteration:
                break

            # decode output
            output_tok = row[1]
            output_str = (
                ""
                if output_tok == self.silence_token
                else self.tokenizer.decode([output_tok])
            )

            with self.state.lock:
                self.state.output_history.append(self._decode(output_tok))
                self.state.row = row_idx
                # thinking history
                for i, c in enumerate(range(2, C)):
                    s = self._decode(row[c])
                    self.state.think_history[i].append(s)
                # user token tracking
                self.state.user_history.append(
                    user_str if user_str is not None else "-"
                )
                # rebuild queued list from what's in the queue
                self.state.user_queued = list(self.token_queue.queue)

                # conversation log: detect turn boundaries
                user_active = user_str is not None
                output_active = output_tok != self.silence_token
                if user_active:
                    if self.state._cur_speaker != "user":
                        # flush previous turn
                        if self.state._cur_speaker and self.state._cur_text.strip():
                            self.state.conv_log.append(
                                (self.state._cur_speaker, self.state._cur_text.strip())
                            )
                        self.state._cur_speaker = "user"
                        self.state._cur_text = ""
                    self.state._cur_text += user_str + " "
                elif output_active:
                    if self.state._cur_speaker != "output":
                        if self.state._cur_speaker and self.state._cur_text.strip():
                            self.state.conv_log.append(
                                (self.state._cur_speaker, self.state._cur_text.strip())
                            )
                        self.state._cur_speaker = "output"
                        self.state._cur_text = ""
                    self.state._cur_text += output_str

            elapsed = time.monotonic() - t0
            sleep = self.tick - elapsed
            if sleep > 0:
                self.stop.wait(sleep)


# ── curses UI ─────────────────────────────────────────────────────────────────
def _safe(win, y, x, text, *attrs):
    """addnstr that silently ignores out-of-bounds."""
    try:
        h, w = win.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            win.addnstr(y, x, text, w - x - 1, *attrs)
    except curses.error:
        pass


def run_ui(stdscr, model, tokenizer, silence_token, args):
    curses.curs_set(1)
    stdscr.nodelay(True)
    stdscr.timeout(33)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # output
    curses.init_pair(2, curses.COLOR_CYAN, -1)  # thinking
    curses.init_pair(3, curses.COLOR_BLUE, -1)  # user consumed
    curses.init_pair(4, curses.COLOR_YELLOW, -1)  # user queued
    curses.init_pair(5, curses.COLOR_WHITE, -1)  # typing

    state = State()
    token_queue = queue.Queue()
    runner = ModelRunner(model, tokenizer, silence_token, state, token_queue, args)

    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()

    input_buf = ""

    try:
        while True:
            # ── handle input ──
            try:
                ch = stdscr.get_wch()
            except curses.error:
                ch = None
            except KeyboardInterrupt:
                break

            if ch is not None:
                if ch == "\x1b" or ch == 27:
                    with state.lock:
                        state.paused = not state.paused
                elif ch in ("\n", "\r", curses.KEY_ENTER):
                    if input_buf.strip():
                        toks = _tokenize_user(tokenizer, input_buf.strip())
                        for t in toks:
                            token_queue.put((t, tokenizer.decode([t]).strip()))
                        with state.lock:
                            state.paused = False
                    input_buf = ""
                elif ch in ("\x7f", "\b", curses.KEY_BACKSPACE, 263):
                    input_buf = input_buf[:-1]
                elif isinstance(ch, str) and ch == " ":
                    if input_buf.strip():
                        toks = _tokenize_user(tokenizer, input_buf.strip())
                        for t in toks:
                            token_queue.put((t, tokenizer.decode([t]).strip()))
                    input_buf = ""
                elif isinstance(ch, str) and ch.isprintable():
                    input_buf += ch

            # ── draw ──
            h, w = stdscr.getmaxyx()
            if h < 10 or w < 50:
                stdscr.erase()
                _safe(stdscr, 0, 0, "Terminal too small (need 50x10+)")
                stdscr.refresh()
                continue

            stdscr.erase()

            # layout
            think_h = 10  # user row + 8 thinking channels + output row
            input_h = 1  # typing line
            dividers = 2
            header_h = 1
            output_h = max(1, h - header_h - think_h - input_h - dividers)

            # ── header ──
            with state.lock:
                row_num = state.row
                paused = state.paused
                qsz = len(state.user_queued)
            if paused:
                status = f"PAUSED  row {row_num}  q:{qsz}  [Esc] resume  [^C] quit"
            else:
                status = f"row {row_num}  {1 / args.tick:.1f} r/s  q:{qsz}  [Esc] pause"
            title = " Stream LLM"
            hdr = f"{title}{status:>{w - len(title) - 1}}"
            stdscr.attron(curses.A_REVERSE)
            _safe(stdscr, 0, 0, hdr.ljust(w))
            stdscr.attroff(curses.A_REVERSE)

            # ── conversation log ──
            with state.lock:
                conv = list(state.conv_log)
                # include the in-progress turn
                if state._cur_speaker and state._cur_text.strip():
                    conv = conv + [(state._cur_speaker, state._cur_text.strip())]
            # wrap each message and collect display lines
            conv_lines = []
            for speaker, text in conv:
                if speaker == "user":
                    prefix = "You: "
                    color = curses.color_pair(3) | curses.A_BOLD
                else:
                    prefix = "LLM: "
                    color = curses.color_pair(1) | curses.A_BOLD
                wrapped = textwrap.wrap(prefix + text, width=w - 4)
                for line in wrapped:
                    conv_lines.append((line, color))
            # show most recent lines that fit
            conv_h = output_h
            visible = conv_lines[-conv_h:] if len(conv_lines) > conv_h else conv_lines
            for i, (line, color) in enumerate(visible):
                y = header_h + (conv_h - len(visible)) + i
                _safe(stdscr, y, 2, line, color)

            # ── thinking divider ──
            think_y = header_h + conv_h
            _safe(stdscr, think_y, 0, ("─ Channels " + "─" * w)[:w], curses.A_DIM)

            # ── aligned cell grid: user + 8 thinking channels + output ──
            with state.lock:
                histories = [list(dq) for dq in state.think_history]
                user_hist = list(state.user_history)
                output_hist = list(state.output_history)
            avail_w = w - 6  # after "Usr: " / "Ana: " label
            cell_w = 10  # fixed cell width for readability
            n_cells = min(30, avail_w // cell_w)

            # user row first
            y = think_y + 1
            _safe(stdscr, y, 1, "Usr:", curses.color_pair(3) | curses.A_BOLD)
            visible = user_hist[-n_cells:] if len(user_hist) > n_cells else user_hist
            offset = n_cells - len(visible)
            for j, tok in enumerate(visible):
                col = 6 + (offset + j) * cell_w
                cell_text = tok[: cell_w - 1].center(cell_w - 1)
                _safe(stdscr, y, col, cell_text, curses.color_pair(3) | curses.A_BOLD)

            # thinking channel rows
            for i, (name, hist) in enumerate(zip(THINK_NAMES, histories)):
                y = think_y + 2 + i
                if y >= h - input_h - 1:
                    break
                _safe(stdscr, y, 1, f"{name}:", curses.color_pair(2) | curses.A_BOLD)
                visible = hist[-n_cells:] if len(hist) > n_cells else hist
                offset = n_cells - len(visible)
                for j, tok in enumerate(visible):
                    col = 6 + (offset + j) * cell_w
                    cell_text = tok[: cell_w - 1].center(cell_w - 1)
                    _safe(
                        stdscr, y, col, cell_text, curses.color_pair(2) | curses.A_BOLD
                    )

            # output cell row
            y = think_y + 2 + len(THINK_NAMES)
            if y < h - input_h:
                _safe(stdscr, y, 1, "Out:", curses.color_pair(1) | curses.A_BOLD)
                visible = (
                    output_hist[-n_cells:] if len(output_hist) > n_cells else output_hist
                )
                offset = n_cells - len(visible)
                for j, tok in enumerate(visible):
                    col = 6 + (offset + j) * cell_w
                    cell_text = tok[: cell_w - 1].center(cell_w - 1)
                    _safe(
                        stdscr, y, col, cell_text, curses.color_pair(1) | curses.A_BOLD
                    )

            # ── typing line ──
            prompt = f" You: {input_buf}"
            _safe(stdscr, h - 1, 0, prompt, curses.color_pair(5) | curses.A_BOLD)

            # cursor
            try:
                stdscr.move(h - 1, min(len(prompt), w - 1))
            except curses.error:
                pass

            stdscr.refresh()

    finally:
        runner.stop.set()


def main():
    parser = argparse.ArgumentParser(description="Interactive stream demo")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tick", type=float, default=1.0, help="Seconds between model steps"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--silence-penalty", type=float, default=10.0)
    parser.add_argument("--think-silence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--arch", default="auto", choices=["auto", "qwen3", "qwen3_5", "qwen3_5_moe"]
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--skip-silence",
        action="store_true",
        help="Mask silence tokens as keys (they don't contribute attention)",
    )
    parser.add_argument(
        "--system-prompt",
        action="store_true",
        help="Warm-start with 10-row system prompt header before user input",
    )
    parser.add_argument(
        "--system-prompt-version",
        default="new",
        choices=["new", "old"],
        help="System prompt variant to use with --system-prompt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map for multi-GPU (e.g. 'auto')",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer, silence_token = load_model(
        args.model, args.device, arch=args.arch, device_map=args.device_map
    )
    print(f"Silence token ID: {silence_token}")
    print("Ready. Launching UI...")

    curses.wrapper(lambda stdscr: run_ui(stdscr, model, tokenizer, silence_token, args))


if __name__ == "__main__":
    main()
