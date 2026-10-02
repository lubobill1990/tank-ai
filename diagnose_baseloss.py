"""Run the pure executor on failing stages; dump the final seconds before
base loss (enemy/bullet geometry) for tactical analysis."""
import json
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
from heuristic import HeuristicBot
from llm_player import execute_order
from mpc_teacher import CachedParser, MpcVetoTeacher


def snapshot(state):
    return {
        'player': None if state.player is None else
            [state.player.x, state.player.y, int(state.player.facing)],
        'enemies': [[e.x, e.y, int(e.facing), e.armor] for e in state.enemies],
        'bullets': [[b.x, b.y, b.vx, b.vy, b.owner] for b in state.bullets],
        'remaining': state.remaining_enemies,
        'lives': state.lives,
    }


def run(stage, seed, max_frames=12000):
    from stable_baselines3.common.utils import set_random_seed
    from rust_train import ROMPrimitiveEnv
    env = ROMPrimitiveEnv(stage, max_frames)
    raw = env.env
    bot = HeuristicBot()
    parser = CachedParser()
    planner = MpcVetoTeacher(depth=12, plan_seeds=3, plan_rng_seed=seed)
    order = {'mode': 'hunt'}
    history = deque(maxlen=45)  # last 3 seconds of decisions
    set_random_seed(seed)
    env.reset(seed=seed)
    outcome = 'unfinished'
    for step in range(max_frames // 4 + 4):
        ram = np.array(raw.nes.ram, dtype=np.uint8, copy=True)
        state = parser.parse(ram)
        history.append({'frame': step * 4, **snapshot(state)})
        masks = np.asarray(raw.action_masks()[:10], dtype=bool)
        action = execute_order(bot, state, order)
        if not masks[action]:
            action = 0
        try:
            scores, unsafe = planner.plan(ram[None])
            action = int(planner.choose(scores, unsafe, [masks], [action])[0])
        except Exception:
            pass
        _, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            outcome = info.get('outcome')
            break
    env.close()
    return outcome, list(history)


def main():
    out = Path('runs/baseloss-diagnosis')
    out.mkdir(parents=True, exist_ok=True)
    for stage in (7, 20, 29, 32, 34):
        for seed in (300, 301):
            outcome, history = run(stage, 260000 + stage * 10 + seed)
            path = out / f'stage{stage:02d}-seed{seed}-{outcome}.json'
            path.write_text(json.dumps(history), encoding='utf-8')
            # concise analysis line: last snapshot's threats
            last = history[-1]
            base_bullets = [b for b in last['bullets']
                            if b[4] >= 2 and abs(b[0] - 120) < 24 and b[1] > 180]
            near = [e for e in last['enemies'] if abs(e[0] - 120) + abs(e[1] - 216) < 90]
            print(json.dumps({
                'stage': stage, 'outcome': outcome, 'final_frame': last['frame'],
                'player': last['player'], 'near_base_enemies': near,
                'base_area_bullets': base_bullets, 'remaining': last['remaining'],
            }), flush=True)


if __name__ == '__main__':
    main()
