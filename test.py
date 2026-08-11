#!/usr/bin/env python3
"""
Ollama 체스 — 로컬 LLM(exaone3.5)과 두는 체스

사용법:
    uv run chess_ollama.py              # exaone3.5와 대국
    uv run chess_ollama.py --black      # 흑을 잡고 두기
    uv run chess_ollama.py --dry-run    # LLM 없이 랜덤 상대와 테스트
    uv run chess_ollama.py --model qwen3:8b

필요 패키지:
    uv add chess ollama
"""
from __future__ import annotations

import argparse
import random
import re
import sys

import chess

MODEL_DEFAULT = "exaone3.5:7.8b"
MAX_RETRY = 3
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "당신은 체스 엔진입니다. 주어진 합법 수 목록에서 가장 좋은 수를 하나만 고릅니다. "
    "목록에 없는 수는 절대 만들어내지 않습니다. 답변은 지정된 형식만 사용합니다."
)


# ─────────────────────────────── 화면 출력 ───────────────────────────────

def render(board: chess.Board, flip: bool = False, ascii_mode: bool = False) -> str:
    """보드를 좌표와 함께 문자열로 그립니다."""
    ranks = range(7, -1, -1) if not flip else range(8)
    files = range(8) if not flip else range(7, -1, -1)

    lines = []
    for r in ranks:
        cells = []
        for f in files:
            piece = board.piece_at(chess.square(f, r))
            if piece is None:
                cells.append("·")
            elif ascii_mode:
                cells.append(piece.symbol())
            else:
                # 어두운 터미널에서는 채워진 글리프가 백(白)으로 잘 보입니다
                cells.append(piece.unicode_symbol(invert_color=True))
        lines.append(f" {r + 1} " + " ".join(cells))

    labels = "abcdefgh" if not flip else "hgfedcba"
    lines.append("   " + " ".join(labels))
    return "\n".join(lines)


def show(board: chess.Board, flip: bool, ascii_mode: bool, note: str = "") -> None:
    print()
    print(render(board, flip, ascii_mode))
    if note:
        print(f"\n  {note}")
    if board.is_check():
        print("  ⚠ 체크!")
    print()


def history_san(board: chess.Board, last_n: int = 12) -> str:
    """지금까지의 기보를 SAN으로 되돌려 만듭니다."""
    replay = chess.Board()
    moves = []
    for mv in board.move_stack:
        moves.append(replay.san(mv))
        replay.push(mv)
    if not moves:
        return "(첫 수)"
    return " ".join(moves[-last_n:])


# ─────────────────────────────── LLM 상대 ───────────────────────────────

def build_prompt(board: chess.Board, legal_sans: list[str], error: str = "") -> str:
    color = "백(White)" if board.turn == chess.WHITE else "흑(Black)"
    listing = "  ".join(legal_sans)

    prompt = f"""당신은 {color}을(를) 두고 있습니다.

현재 국면 (FEN):
{board.fen()}

기보:
{history_san(board)}

선택 가능한 합법 수 (이 목록에 있는 것만 사용):
{listing}

위 목록에서 가장 좋은 수를 정확히 하나 골라 아래 형식으로만 답하세요.

MOVE: <목록의 수를 그대로 복사>
WHY: <한국어 한 문장 설명>"""

    if error:
        prompt += f"\n\n[이전 답변 오류] {error} 다시 목록에서만 고르세요."
    return prompt


def parse_move(text: str, board: chess.Board, legal_sans: list[str]):
    """LLM 응답에서 합법 수를 추출합니다. 실패하면 (None, 사유)."""
    # <think> 같은 추론 블록 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    m = re.search(r"MOVE\s*[:：]\s*(\S+)", text, re.IGNORECASE)
    candidate = m.group(1).strip().strip("`*'\".,") if m else ""

    # 1순위: MOVE 줄을 SAN으로 해석
    for token in ([candidate] if candidate else []):
        try:
            return board.parse_san(token), ""
        except ValueError:
            pass
        try:
            mv = chess.Move.from_uci(token.lower())
            if mv in board.legal_moves:
                return mv, ""
        except ValueError:
            pass

    # 2순위: 응답 전체에서 합법 SAN 문자열을 탐색 (긴 것부터 = 오탐 방지)
    for san in sorted(legal_sans, key=len, reverse=True):
        if re.search(rf"(?<![\w+#=-]){re.escape(san)}(?![\w+#=])", text):
            return board.parse_san(san), ""

    return None, f"'{candidate or text.strip()[:30]}'은(는) 합법 수가 아닙니다."


def llm_move(board: chess.Board, model: str):
    """LLM에게 수를 물어봅니다. 실패 시 랜덤 합법 수로 대체."""
    import ollama

    legal_sans = [board.san(mv) for mv in board.legal_moves]
    error = ""

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(board, legal_sans, error)},
                ],
                options={"temperature": TEMPERATURE, "num_predict": 150},
            )
        except Exception as e:
            print(f"  ✖ Ollama 호출 실패: {e}")
            print("    → 'ollama serve'가 실행 중인지, 모델 이름이 맞는지 확인하세요.")
            sys.exit(1)

        try:
            text = resp["message"]["content"]
        except (TypeError, KeyErㅃror):
            text = resp.message.content

        move, error = parse_move(text, board, legal_sans)
        if move:
            why = re.search(r"WHY\s*[:：]\s*(.+)", text)
            return move, (why.group(1).strip()[:80] if why else "")

        print(f"  … 응답 재요청 {attempt}/{MAX_RETRY} ({error})")

    fallback = random.choice(list(board.legal_moves))
    return fallback, "(모델이 합법 수를 못 내서 임의 수로 대체)"


def random_move(board: chess.Board, model: str):
    """--dry-run용 더미 상대."""
    return random.choice(list(board.legal_moves)), "(랜덤 상대)"


# ─────────────────────────────── 사람 입력 ───────────────────────────────

HELP = """
  명령어
    e4, Nf3, O-O   기보(SAN) 표기로 두기
    e2e4           좌표(UCI) 표기로 두기
    moves          지금 둘 수 있는 수 보기
    undo           내 수와 상대 수를 한 턴 되돌리기
    fen            현재 국면 FEN 출력
    quit           종료
"""


def human_move(board: chess.Board):
    """사람의 한 수를 받습니다. ('move', Move) 또는 ('undo'/'quit', None)."""
    while True:
        try:
            raw = input("  당신의 수 > ").strip()
        except (EOFError, KeyboardInterrupt):
            return "quit", None

        if not raw:
            continue
        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            return "quit", None
        if cmd == "undo":
            return "undo", None
        if cmd == "help":
            print(HELP)
            continue
        if cmd == "fen":
            print(f"  {board.fen()}")
            continue
        if cmd == "moves":
            sans = [board.san(mv) for mv in board.legal_moves]
            print("  " + "  ".join(sorted(sans)))
            continue

        try:
            return "move", board.parse_san(raw)
        except ValueError:
            pass
        try:
            mv = chess.Move.from_uci(cmd)
            if mv in board.legal_moves:
                return "move", mv
        except ValueError:
            pass

        print("  ✖ 둘 수 없는 수입니다. 'moves'로 목록을 확인하세요.")


# ─────────────────────────────── 대국 진행 ───────────────────────────────

def result_text(board: chess.Board, human_color: bool) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "대국 중단"
    if outcome.winner is None:
        reasons = {
            chess.Termination.STALEMATE: "스테일메이트",
            chess.Termination.INSUFFICIENT_MATERIAL: "기물 부족",
            chess.Termination.FIFTY_MOVES: "50수 규칙",
            chess.Termination.THREEFOLD_REPETITION: "3회 동형 반복",
        }
        return f"무승부 ({reasons.get(outcome.termination, outcome.termination.name)})"
    who = "당신" if outcome.winner == human_color else "AI"
    how = "체크메이트" if outcome.termination == chess.Termination.CHECKMATE else outcome.termination.name
    return f"{who}의 승리! ({how})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ollama와 두는 체스")
    ap.add_argument("--model", default=MODEL_DEFAULT, help=f"모델 이름 (기본 {MODEL_DEFAULT})")
    ap.add_argument("--black", action="store_true", help="흑을 잡습니다 (AI 선공)")
    ap.add_argument("--ascii", action="store_true", help="유니코드 대신 알파벳으로 표시")
    ap.add_argument("--dry-run", action="store_true", help="LLM 없이 랜덤 상대와 테스트")
    args = ap.parse_args()

    human_color = chess.BLACK if args.black else chess.WHITE
    engine = random_move if args.dry_run else llm_move
    board = chess.Board()
    flip = args.black

    opponent = "랜덤 상대" if args.dry_run else args.model
    print(f"\n♟  Ollama 체스 — 상대: {opponent}")
    print(f"   당신: {'흑' if args.black else '백'}   |   'help'로 명령어 확인")
    show(board, flip, args.ascii)

    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            action, move = human_move(board)
            if action == "quit":
                print("\n  대국을 종료합니다.\n")
                return
            if action == "undo":
                if len(board.move_stack) < 2:
                    print("  ✖ 되돌릴 수가 없습니다.")
                    continue
                board.pop()
                board.pop()
                show(board, flip, args.ascii, "한 턴 되돌렸습니다.")
                continue
            san = board.san(move)
            board.push(move)
            show(board, flip, args.ascii, f"당신: {san}")
        else:
            print(f"  🤔 {opponent} 생각 중...")
            move, why = engine(board, args.model)
            san = board.san(move)
            board.push(move)
            note = f"AI: {san}" + (f"  —  {why}" if why else "")
            show(board, flip, args.ascii, note)

    print("─" * 40)
    print(f"  {result_text(board, human_color)}")
    print(f"  총 {board.fullmove_number - 1}수")
    print("─" * 40 + "\n")


if __name__ == "__main__":
    main()