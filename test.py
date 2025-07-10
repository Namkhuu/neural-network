import tensorflow as tf
import chess
import chess.pgn
import chess.engine
import numpy as np
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input, Dense

# ===== Board to State =====
def board_to_state(board):
    state = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            idx = piece_map[piece.piece_type] + color_offset
            row, col = divmod(square, 8)
            state[row][col][idx] = 1
    return state.flatten()

# ===== All Possible UCI Moves =====
def all_possible_moves():
    squares = [chess.square_name(i) for i in range(64)]
    promotions = ['q', 'r', 'b', 'n']
    normal_moves = [a + b for a in squares for b in squares]
    promo_moves = [a + b + p for a in squares for b in squares for p in promotions]
    return sorted(set(normal_moves + promo_moves))

ALL_MOVES = all_possible_moves()
MOVE_IDX = {move: i for i, move in enumerate(ALL_MOVES)}

# ===== Reward Function =====
def get_intermediate_reward(prev_board, new_board):
    reward = 0
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    def material_score(b):
        score = 0
        for p, v in piece_values.items():
            score += len(b.pieces(p, chess.WHITE)) * v
            score -= len(b.pieces(p, chess.BLACK)) * v
        return score

    reward += material_score(new_board) - material_score(prev_board)
    if new_board.is_check():
        reward += 0.3
    return reward

# ===== One-step Rollout Evaluation =====
def rollout_value(board, depth=1):
    temp_board = board.copy()
    for _ in range(depth):
        if temp_board.is_game_over():
            break
        legal_moves = list(temp_board.legal_moves)
        if not legal_moves:
            break
        temp_board.push(np.random.choice(legal_moves))
    temp_state = board_to_state(temp_board)
    temp_tensor = tf.convert_to_tensor([temp_state], dtype=tf.float32)
    return critic_model(temp_tensor)[0][0].numpy()

# ===== Load Models If Available =====
try:
    actor_model = load_model("actor_model.keras")
    critic_model = load_model("critic_model.keras")
    print("Loaded saved models.")
except:
    # ===== Build Models If Not Found =====
    actor_input = Input(shape=(768,))
    actor_hidden = Dense(128, activation='relu')(actor_input)
    actor_output = Dense(len(ALL_MOVES), activation='softmax')(actor_hidden)
    actor_model = Model(inputs=actor_input, outputs=actor_output)

    critic_input = Input(shape=(768,))
    critic_hidden = Dense(128, activation='relu')(critic_input)
    critic_output = Dense(1, activation='linear')(critic_hidden)
    critic_model = Model(inputs=critic_input, outputs=critic_output)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

gamma = 0.99
num_episodes = 10000

# ===== Start Stockfish Engine =====
engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
engine.configure({"Skill Level": 1})  # Weakest level

# ===== Training Loop =====
for ep in range(num_episodes):
    board = chess.Board()
    total_reward = 0
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # ===== Agent's Move (with rollout evaluation) =====
            move_scores = []
            for move in board.legal_moves:
                sim_board = board.copy()
                sim_board.push(move)
                value = rollout_value(sim_board, depth=1)
                move_scores.append((move, value))

            best_move = max(move_scores, key=lambda x: x[1])[0]
            prev_board = board.copy()
            board.push(best_move)
            node = node.add_variation(best_move)

            reward = get_intermediate_reward(prev_board, board)
            if board.is_game_over():
                result = board.result()
                final_reward = 1 if result == "1-0" else -1 if result == "0-1" else 0
                reward += final_reward

            state = board_to_state(prev_board)
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            next_state = board_to_state(board)
            next_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
            value = critic_model(state_tensor)[0][0].numpy()
            next_value = critic_model(next_tensor)[0][0].numpy()

            td_target = reward + gamma * (1 - board.is_game_over()) * next_value
            advantage = td_target - value

            # === Train Actor ===
            with tf.GradientTape() as tape:
                probs = actor_model(state_tensor)[0]
                move_idx = MOVE_IDX[best_move.uci()] if best_move.uci() in MOVE_IDX else 0
                log_prob = tf.math.log(probs[move_idx] + 1e-8)
                actor_loss = -log_prob * advantage
            grads = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))

            # === Train Critic ===
            with tf.GradientTape() as tape:
                v = critic_model(state_tensor)[0][0]
                critic_loss = tf.square(td_target - v)
            grads = tape.gradient(critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_model.trainable_variables))

            total_reward += reward
        else:
            # ===== Stockfish's Move =====
            result = engine.play(board, chess.engine.Limit(depth=1))
            board.push(result.move)
            node = node.add_variation(result.move)

    print(f"Episode {ep+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Result: {board.result()}")

    with open(f"games/pgn_episode_{ep+1}.pgn", "w") as f:
        print(game, file=f)

engine.quit()
actor_model.save("actor_model.keras")
critic_model.save("critic_model.keras")
