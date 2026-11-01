"""Play against the one-hidden-layer chess-style model.

Copy this file beside encoding.py and engine_cache.py, then run:
    py play_chess.py
Optional: --engine PATH --model PATH
No additional pip dependencies beyond numpy and python-chess.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter.font as tkfont

import chess
import chess.engine
import chess.pgn
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'chess_data' / 'ollervo'
BG, PANEL, INK, MUTED, ACCENT = '#141c24', '#1e2935', '#ecf1ed', '#a8b8c4', '#a5d6b0'
LIGHT, DARK = '#ebead5', '#729182'
PIECES = {'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
          'k': '\u265a', 'q': '\u265b', 'r': '\u265c', 'b': '\u265d', 'n': '\u265e', 'p': '\u265f'}


def load_weights(path):
    """Validate the exact checkpoint format produced by train_engine.py."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'W1', 'b1', 'W2', 'b2'}:
            raise ValueError('Expected W1, b1, W2 and b2 from the one-hidden-layer model.')
        weights = {name: np.array(archive[name], dtype=np.float32) for name in archive.files}
    w1, b1, w2, b2 = (weights[name] for name in ('W1', 'b1', 'W2', 'b2'))
    if (w1.ndim != 2 or w1.shape[0] != 1353 or w1.shape[1] < 1
            or b1.shape != (w1.shape[1],) or w2.shape != (w1.shape[1], 1)
            or b2.shape != (1,)):
        raise ValueError('Checkpoint must use 1,353 input features. Select best_engine_model.npz.')
    if not all(np.isfinite(value).all() for value in weights.values()):
        raise ValueError('Checkpoint contains non-finite weights.')
    return weights


def predict(inputs, weights):
    hidden = np.maximum(0, inputs @ weights['W1'] + weights['b1'])
    scores = (hidden @ weights['W2'] + weights['b2']).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError('The model produced non-finite scores.')
    probabilities = np.exp(scores - scores.max()).astype(np.float64)
    probabilities /= probabilities.sum()
    return probabilities


def choose_move(probabilities, mode, rng):
    if mode == 'Most likely':
        return int(np.argmax(probabilities))
    return int(rng.choice(len(probabilities), p=probabilities))


def screen_square(row, col, flipped):
    return chess.square(7 - col, row) if flipped else chess.square(col, 7 - row)


def history_lines(board):
    replay = board.root()
    lines = []
    for move in board.move_stack:
        san = replay.san(move)
        if replay.turn == chess.WHITE:
            lines.append(f'{replay.fullmove_number:>3}.  {san}')
        elif lines:
            lines[-1] += f'    {san}'
        else:
            lines.append(f'{replay.fullmove_number:>3}... {san}')
        replay.push(move)
    return '\n'.join(lines) or 'Your game starts here.'


class ChessApp:
    def __init__(self, root, engine_path='', model_path=''):
        self.root = root
        root.title('Ollervo / Play your model')
        root.configure(bg=BG)
        root.geometry('1120x850')
        root.minsize(940, 760)
        self.board = chess.Board()
        self.human = chess.WHITE
        self.flipped = False
        self.selected = None
        self.busy = False
        self.started = False
        self.finished = None
        self.game_result = None
        self.generation = 0
        self.outbox = queue.Queue()
        self.worker = None
        self.engine_lock = threading.Lock()
        self.active_engine = None
        self.closing = False
        self.margin, self.cell = 24, 72
        self.rng = np.random.default_rng()
        settings = {}
        try:
            settings = json.loads((DATA / 'ui_settings.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
        self.engine_path = tk.StringVar(value=engine_path or settings.get('engine', ''))
        self.model_path = tk.StringVar(value=model_path or settings.get('model', str(ROOT / 'models' / 'best_engine_model.npz')))
        self.side = tk.StringVar(value='White')
        self.mode = tk.StringVar(value='Sample probabilities')
        self.status = tk.StringVar(value='Choose Stockfish and your checkpoint, then start a game.')
        self.detail = tk.StringVar(value='Depth 10 / one Stockfish thread / original model probabilities')
        self.build_ui()
        self.draw()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(80, self.poll)

    def build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground=PANEL, background=PANEL, foreground=INK,
                        arrowcolor=ACCENT, padding=5)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', INK)])
        tk.Label(self.root, text='OLLERVO  /  CHESS LAB',
                 font=('Segoe UI', 10, 'bold'), fg=ACCENT, bg=BG).pack(anchor='w', padx=24, pady=(18, 0))
        tk.Label(self.root, text='Play your model.', font=('Segoe UI', 25, 'bold'),
                 fg=INK, bg=BG).pack(anchor='w', padx=24, pady=(2, 12))
        setup = tk.Frame(self.root, bg=PANEL, padx=12, pady=10)
        setup.pack(fill='x', padx=24)
        for row, (label, variable, kind) in enumerate([
            ('Stockfish', self.engine_path, 'engine'), ('Checkpoint', self.model_path, 'model')
        ]):
            tk.Label(setup, text=label, bg=PANEL, fg=MUTED, width=11, anchor='w').grid(row=row, column=0)
            tk.Entry(setup, textvariable=variable, bg=BG, fg=INK, insertbackground=INK,
                     relief='flat', font=('Segoe UI', 10)).grid(row=row, column=1, sticky='ew', padx=8, pady=4, ipady=5)
            self.button(setup, 'Browse', lambda k=kind: self.browse(k)).grid(row=row, column=2)
        setup.columnconfigure(1, weight=1)
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=24, pady=(14, 10))
        left = tk.Frame(body, bg=BG)
        left.pack(side='left', fill='both', expand=True)
        right = tk.Frame(body, bg=PANEL, padx=16, pady=14, width=290)
        right.pack(side='right', fill='y', padx=(18, 0))
        right.pack_propagate(False)
        self.canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.canvas.bind('<Button-1>', self.click)
        tk.Label(left, textvariable=self.status, bg=BG, fg=INK, anchor='w',
                 justify='left', wraplength=610, font=('Segoe UI', 11)).pack(fill='x', pady=(8, 2))
        tk.Label(left, textvariable=self.detail, bg=BG, fg=MUTED, anchor='w',
                 font=('Segoe UI', 9)).pack(fill='x')
        tk.Label(right, text='YOUR SEAT', fg=ACCENT, bg=PANEL, font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        self.side_box = ttk.Combobox(right, textvariable=self.side, values=['White', 'Black'], state='readonly')
        self.side_box.pack(fill='x', pady=(5, 12))
        tk.Label(right, text='OPPONENT CHOICE', fg=ACCENT, bg=PANEL, font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        self.mode_box = ttk.Combobox(right, textvariable=self.mode,
                                   values=['Sample probabilities', 'Most likely'], state='readonly')
        self.mode_box.pack(fill='x', pady=(5, 12))
        self.start_button = self.button(right, 'New game', self.new_game, primary=True)
        self.start_button.pack(fill='x', pady=(0, 8))
        row = tk.Frame(right, bg=PANEL)
        row.pack(fill='x')
        self.button(row, 'Undo turn', self.undo).pack(side='left', expand=True, fill='x', padx=(0, 4))
        self.button(row, 'Flip board', self.flip).pack(side='left', expand=True, fill='x', padx=(4, 0))
        row = tk.Frame(right, bg=PANEL)
        row.pack(fill='x', pady=8)
        self.button(row, 'Load FEN', self.load_fen).pack(side='left', expand=True, fill='x', padx=(0, 4))
        self.button(row, 'Copy FEN', self.copy_fen).pack(side='left', expand=True, fill='x', padx=(4, 0))
        self.button(right, 'Save game / PGN', self.save_pgn).pack(fill='x', pady=(0, 8))
        self.button(right, 'Claim draw', self.claim_draw).pack(fill='x', pady=(0, 14))
        tk.Label(right, text='LAST MODEL DECISION', fg=ACCENT, bg=PANEL,
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        self.probs = tk.Text(right, height=7, bg=PANEL, fg=INK, relief='flat', state='disabled',
                             font=('Consolas', 11), cursor='arrow', wrap='none')
        self.probs.pack(fill='x', pady=(5, 12))
        self.write_text(self.probs, 'Model probabilities appear\nafter its first move.')
        tk.Label(right, text='MOVES', fg=ACCENT, bg=PANEL, font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        self.history = tk.Text(right, height=6, bg=BG, fg=INK, relief='flat', state='disabled',
                               font=('Consolas', 11), wrap='word', padx=8, pady=8)
        self.history.pack(fill='both', expand=True, pady=(6, 0))
        families = set(tkfont.families())
        self.piece_font = next((font for font in ['Segoe UI Symbol', 'DejaVu Sans', 'Arial Unicode MS']
                               if font in families), 'TkDefaultFont')

    @staticmethod
    def button(parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, relief='flat', bd=0,
                         bg=ACCENT if primary else '#304252', fg=BG if primary else INK,
                         activebackground='#c5e8cc', activeforeground=BG, cursor='hand2',
                         padx=10, pady=7, font=('Segoe UI', 10, 'bold' if primary else 'normal'))

    @staticmethod
    def write_text(widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def browse(self, kind):
        path = filedialog.askopenfilename(title='Select Stockfish' if kind == 'engine' else 'Select model checkpoint',
                                         filetypes=[('Executable', '*.exe'), ('All files', '*')] if kind == 'engine'
                                         else [('NumPy checkpoint', '*.npz'), ('All files', '*')])
        if path:
            (self.engine_path if kind == 'engine' else self.model_path).set(path)

    def prepare(self):
        if self.busy:
            messagebox.showinfo('Please wait', 'Wait for the current model move to finish.')
            return False
        try:
            engine = Path(self.engine_path.get()).expanduser().resolve()
            if not engine.is_file():
                raise ValueError('Browse to your working Stockfish executable first.')
            self.weights = load_weights(Path(self.model_path.get()).expanduser())
            self.engine_file = str(engine)
            self.choice_mode = self.mode.get()
            self.human = chess.WHITE if self.side.get() == 'White' else chess.BLACK
            DATA.mkdir(parents=True, exist_ok=True)
            (DATA / 'ui_settings.json').write_text(json.dumps({
                'engine': self.engine_file, 'model': self.model_path.get()
            }), encoding='utf-8')
            self.flipped = self.human == chess.BLACK
            return True
        except Exception as error:
            messagebox.showerror('Could not start', str(error))
            return False

    def start(self, board):
        self.generation += 1
        self.board = board
        self.selected = None
        self.finished = None
        self.game_result = None
        self.started = True
        self.write_text(self.probs, 'Waiting for the model\'s next turn.')
        self.update_game()

    def new_game(self):
        if self.prepare():
            self.start(chess.Board())

    def load_fen(self):
        if self.busy:
            return
        text = simpledialog.askstring('Load a position', 'Paste a standard-chess FEN:')
        if text is None:
            return
        try:
            board = chess.Board(text.strip())
            if not board.is_valid():
                raise ValueError('The FEN does not describe a valid standard-chess position.')
        except ValueError as error:
            messagebox.showerror('Invalid FEN', str(error))
            return
        if self.prepare():
            self.start(board)

    def draw(self):
        canvas = self.canvas
        canvas.delete('all')
        size = max(200, min(canvas.winfo_width(), canvas.winfo_height()) - 44)
        self.cell = max(20, size // 8)
        self.margin = 22
        cell, margin = self.cell, self.margin
        targets = set()
        if self.selected is not None:
            targets = {move.to_square for move in self.board.legal_moves if move.from_square == self.selected}
        last = self.board.peek() if self.board.move_stack else None
        checked_king = self.board.king(self.board.turn) if self.board.is_check() else None
        for row in range(8):
            for col in range(8):
                square = screen_square(row, col, self.flipped)
                x, y = margin + col * cell, margin + row * cell
                colour = LIGHT if (row + col) % 2 == 0 else DARK
                if last and square in (last.from_square, last.to_square):
                    colour = '#c8ce80' if (row + col) % 2 == 0 else '#a4b469'
                if square == self.selected:
                    colour = '#e1c46d'
                if square == checked_king:
                    colour = '#d77d70'
                canvas.create_rectangle(x, y, x + cell, y + cell, fill=colour, outline=colour)
                piece = self.board.piece_at(square)
                if piece:
                    # Solid glyphs for both colours, with an outline for white pieces.
                    glyph = PIECES[piece.symbol().lower()]
                    fontsize = max(14, int(cell * 0.62))
                    if piece.color == chess.WHITE:
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            canvas.create_text(x + cell / 2 + dx, y + cell / 2 + dy, text=glyph,
                                               font=(self.piece_font, fontsize), fill='#24352e')
                    canvas.create_text(x + cell / 2, y + cell / 2, text=glyph,
                                       font=(self.piece_font, fontsize),
                                       fill='#fffef2' if piece.color else '#1e2c29')
                if square in targets:
                    if piece or self.board.is_en_passant(chess.Move(self.selected, square)):
                        canvas.create_rectangle(x+3, y+3, x+cell-3, y+cell-3,
                                                outline='#355d4d', width=3)
                    else:
                        radius = cell * 0.075
                        canvas.create_oval(x+cell/2-radius, y+cell/2-radius,
                                           x+cell/2+radius, y+cell/2+radius,
                                           fill='#355d4d', outline='')
        for col in range(8):
            square = screen_square(7, col, self.flipped)
            canvas.create_text(margin + (col+.5)*cell, margin+8*cell+12,
                               text=chess.FILE_NAMES[chess.square_file(square)], fill=MUTED,
                               font=('Segoe UI', 10))
        for row in range(8):
            square = screen_square(row, 0, self.flipped)
            canvas.create_text(10, margin+(row+.5)*cell, text=str(chess.square_rank(square)+1),
                               fill=MUTED, font=('Segoe UI', 10))

    def click(self, event):
        if not self.started or self.busy or self.finished or self.board.turn != self.human:
            return
        row = (event.y - self.margin) // self.cell
        col = (event.x - self.margin) // self.cell
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        square = screen_square(row, col, self.flipped)
        piece = self.board.piece_at(square)
        if self.selected is not None:
            candidates = [move for move in self.board.legal_moves
                          if move.from_square == self.selected and move.to_square == square]
            if candidates:
                move = candidates[0]
                if len(candidates) > 1:
                    move = self.promotion(candidates)
                    if move is None:
                        return
                self.board.push(move)
                self.selected = None
                self.update_game()
                return
        self.selected = square if piece and piece.color == self.human and square != self.selected else None
        self.draw()

    def promotion(self, candidates):
        result = []
        window = tk.Toplevel(self.root)
        window.title('Promote pawn')
        window.configure(bg=PANEL)
        window.transient(self.root)
        window.resizable(False, False)
        tk.Label(window, text='Choose your promotion', bg=PANEL, fg=INK,
                 font=('Segoe UI', 13)).pack(padx=20, pady=12)
        row = tk.Frame(window, bg=PANEL)
        row.pack(padx=16, pady=(0, 16))
        def select(move):
            result.append(move)
            window.destroy()
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            move = next(move for move in candidates if move.promotion == piece_type)
            self.button(row, chess.piece_name(piece_type).title(), lambda m=move: select(m)).pack(side='left', padx=3)
        window.grab_set()
        self.root.wait_window(window)
        return result[0] if result else None

    def update_game(self):
        self.draw()
        self.write_text(self.history, history_lines(self.board))
        self.history.see('end')
        outcome = self.board.outcome()
        if outcome and not self.finished:
            self.game_result = outcome.result()
            self.finished = f'{outcome.result()} / {outcome.termination.name.replace("_", " ").title()}'
        if self.finished:
            self.status.set(self.finished)
        elif self.board.turn == self.human:
            self.status.set('Your move.' + (' You are in check.' if self.board.is_check() else ''))
        else:
            self.begin_model_turn()

    def begin_model_turn(self):
        if self.busy:
            return
        self.busy = True
        self.status.set('Model thinking / analysing all legal moves...')
        self.detail.set('Depth 10 / cached evaluations are reused / board remains responsive')
        self.start_button.configure(state='disabled')
        token, fen = self.generation, self.board.fen()
        weights, mode, engine_file = self.weights, self.choice_mode, self.engine_file
        self.worker = threading.Thread(target=self.calculate,
                                       args=(token, fen, weights, mode, engine_file), daemon=True)
        self.worker.start()

    def calculate(self, token, fen, weights, mode, engine_file):
        """Background work only. Never access a Tk widget in this thread."""
        engine = None
        try:
            from encoding import encode_engine_candidates
            from engine_cache import analyse_cached
            started = time.perf_counter()
            # Match the FEN-only analysis used to prepare training data.
            board = chess.Board(fen)
            connection = sqlite3.connect(DATA / 'engine_cache.sqlite3', timeout=30)
            try:
                engine = chess.engine.SimpleEngine.popen_uci(engine_file)
                with self.engine_lock:
                    if self.closing:
                        engine.close()
                        return
                    self.active_engine = engine
                evaluations, hit = analyse_cached(engine, connection, board, depth=10)
            finally:
                if engine:
                    try:
                        engine.quit()
                    except Exception:
                        engine.close()
                with self.engine_lock:
                    self.active_engine = None
                connection.close()
            moves, inputs = encode_engine_candidates(fen, evaluations)
            probabilities = predict(inputs, weights)
            index = choose_move(probabilities, mode, self.rng)
            selected = moves[index]
            ranking = np.argsort(-probabilities)[:5]
            text = '\n'.join(f'{board.san(moves[i]):9s} {probabilities[i]:6.1%}' for i in ranking)
            text += f'\n\nPlayed {board.san(selected)} ({probabilities[index]:.1%})'
            self.outbox.put((token, selected, text, hit, time.perf_counter()-started, None))
        except Exception as error:
            self.outbox.put((token, None, '', False, 0, str(error)))

    def poll(self):
        try:
            while True:
                token, move, text, hit, elapsed, error = self.outbox.get_nowait()
                if token != self.generation:
                    continue
                self.busy = False
                self.start_button.configure(state='normal')
                if error:
                    self.status.set('Could not play a model move. Check the paths and project files.')
                    messagebox.showerror('Model error', error + '\n\nUse New game to retry after correcting the problem.')
                elif move not in self.board.legal_moves:
                    self.status.set('Discarded an outdated model move. Start a new game.')
                else:
                    self.board.push(move)
                    self.write_text(self.probs, text)
                    self.detail.set(f'{self.choice_mode} / {elapsed:.1f}s / ' + ('cache hit' if hit else 'new analysis'))
                    self.update_game()
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(80, self.poll)

    def undo(self):
        if self.busy or not self.started or not self.board.move_stack:
            return
        self.generation += 1
        self.board.pop()
        while self.board.move_stack and self.board.turn != self.human:
            self.board.pop()
        self.finished = self.game_result = None
        self.selected = None
        self.write_text(self.probs, 'Previous prediction cleared.')
        self.update_game()

    def flip(self):
        self.flipped = not self.flipped
        self.draw()

    def copy_fen(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.board.fen())
        self.detail.set('FEN copied to clipboard.')

    def claim_draw(self):
        if not self.started or self.busy or self.finished or self.board.turn != self.human:
            return
        if self.board.can_claim_draw():
            self.finished = '1/2-1/2 / Draw claimed'
            self.game_result = '1/2-1/2'
            self.update_game()
        else:
            messagebox.showinfo('Draw claim', 'A repetition or fifty-move draw cannot currently be claimed.')

    def save_pgn(self):
        path = filedialog.asksaveasfilename(defaultextension='.pgn', initialfile='vs_ollervo_model.pgn',
                                          filetypes=[('Chess game', '*.pgn')])
        if not path:
            return
        try:
            game = chess.pgn.Game.from_board(self.board)
            game.headers['Event'] = 'Play your model'
            game.headers['Date'] = date.today().strftime('%Y.%m.%d')
            game.headers['White'] = 'You' if self.human else 'Ollervo model'
            game.headers['Black'] = 'Ollervo model' if self.human else 'You'
            game.headers['Result'] = self.game_result or self.board.result()
            Path(path).write_text(str(game) + '\n', encoding='utf-8')
            self.detail.set('Game saved as PGN.')
        except OSError as error:
            messagebox.showerror('Could not save', str(error))

    def close(self):
        self.closing = True
        self.generation += 1
        with self.engine_lock:
            if self.active_engine is not None:
                self.active_engine.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', default='')
    parser.add_argument('--model', default='')
    args = parser.parse_args()
    try:
        import encoding
        import engine_cache
        if not hasattr(encoding, 'encode_engine_candidates'):
            raise ImportError('encoding.py must include encode_engine_candidates().')
        if not hasattr(engine_cache, 'analyse_cached'):
            raise ImportError('engine_cache.py must include analyse_cached().')
    except ImportError as error:
        raise SystemExit(f'{error}\nPlace play_chess.py beside your encoding.py and engine_cache.py.')
    root = tk.Tk()
    ChessApp(root, args.engine, args.model)
    root.mainloop()


if __name__ == '__main__':
    main()
