#include "go_game.h"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace alpha_go {

GoBoard::GoBoard(int size, float komi)
    : size_(size),
      komi_(komi),
      board_(size * size, EMPTY),
      neighbor_indices_(size * size),
      neighbor_counts_(size * size) {
    init_neighbors();
}

void GoBoard::init_neighbors() {
    for (int row = 0; row < size_; ++row) {
        for (int col = 0; col < size_; ++col) {
            int idx = flat_index(row, col);
            int count = 0;

            // Up
            if (row > 0) {
                neighbor_indices_[idx][count++] = flat_index(row - 1, col);
            }
            // Down
            if (row < size_ - 1) {
                neighbor_indices_[idx][count++] = flat_index(row + 1, col);
            }
            // Left
            if (col > 0) {
                neighbor_indices_[idx][count++] = flat_index(row, col - 1);
            }
            // Right
            if (col < size_ - 1) {
                neighbor_indices_[idx][count++] = flat_index(row, col + 1);
            }

            neighbor_counts_[idx] = count;
        }
    }
}

std::pair<std::vector<int>, std::vector<int>> GoBoard::get_group_and_liberties(int index) const {
    int8_t color = board_[index];
    if (color == EMPTY) {
        return {{}, {}};
    }

    std::vector<int> group;
    std::vector<int> liberties;
    std::vector<bool> visited(size_ * size_, false);
    std::vector<bool> liberty_visited(size_ * size_, false);

    std::queue<int> queue;
    queue.push(index);
    visited[index] = true;

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        group.push_back(current);

        for (int i = 0; i < neighbor_counts_[current]; ++i) {
            int neighbor = neighbor_indices_[current][i];

            if (board_[neighbor] == EMPTY) {
                if (!liberty_visited[neighbor]) {
                    liberty_visited[neighbor] = true;
                    liberties.push_back(neighbor);
                }
            } else if (board_[neighbor] == color && !visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    return {group, liberties};
}

int GoBoard::remove_group(const std::vector<int>& group) {
    for (int idx : group) {
        board_[idx] = EMPTY;
    }
    return static_cast<int>(group.size());
}

bool GoBoard::would_be_suicide(int index) const {
    int8_t color = to_play_;
    int8_t opponent = (color == BLACK) ? WHITE : BLACK;

    // Temporarily place the stone
    auto& mutable_board = const_cast<std::vector<int8_t>&>(board_);
    mutable_board[index] = color;

    // Check if our stone has liberties
    auto [group, liberties] = get_group_and_liberties(index);

    if (!liberties.empty()) {
        // Has liberties - not suicide
        mutable_board[index] = EMPTY;
        return false;
    }

    // No direct liberties - check if we capture any opponent stones
    for (int i = 0; i < neighbor_counts_[index]; ++i) {
        int neighbor = neighbor_indices_[index][i];
        if (mutable_board[neighbor] == opponent) {
            auto [opp_group, opp_liberties] = get_group_and_liberties(neighbor);
            if (opp_liberties.empty()) {
                // Would capture opponent - not suicide
                mutable_board[index] = EMPTY;
                return false;
            }
        }
    }

    // No liberties and no captures - suicide
    mutable_board[index] = EMPTY;
    return true;
}

bool GoBoard::is_legal(int row, int col) const {
    return is_legal_flat(flat_index(row, col));
}

bool GoBoard::is_legal_flat(int index) const {
    // Check bounds
    if (index < 0 || index >= size_ * size_) {
        return false;
    }

    // Must be empty
    if (board_[index] != EMPTY) {
        return false;
    }

    // Cannot play on ko point
    if (ko_point_.has_value() && ko_point_.value() == index) {
        return false;
    }

    // Cannot commit suicide
    if (would_be_suicide(index)) {
        return false;
    }

    return true;
}

std::vector<int> GoBoard::get_legal_moves_flat() const {
    std::vector<int> moves;
    moves.reserve(size_ * size_);

    for (int i = 0; i < size_ * size_; ++i) {
        if (is_legal_flat(i)) {
            moves.push_back(i);
        }
    }

    return moves;
}

bool GoBoard::play(int row, int col) {
    return play_flat(flat_index(row, col));
}

bool GoBoard::play_flat(int index) {
    if (!is_legal_flat(index)) {
        return false;
    }

    // Place the stone
    board_[index] = to_play_;
    int8_t opponent = (to_play_ == BLACK) ? WHITE : BLACK;

    // Reset ko point
    ko_point_ = std::nullopt;

    // Check for captures
    int total_captured = 0;
    int last_captured_idx = -1;

    for (int i = 0; i < neighbor_counts_[index]; ++i) {
        int neighbor = neighbor_indices_[index][i];
        if (board_[neighbor] == opponent) {
            auto [group, liberties] = get_group_and_liberties(neighbor);
            if (liberties.empty()) {
                // Capture this group
                if (group.size() == 1) {
                    last_captured_idx = group[0];
                }
                total_captured += remove_group(group);
            }
        }
    }

    // Set ko point if exactly one stone was captured and our stone has exactly one liberty
    if (total_captured == 1) {
        auto [our_group, our_liberties] = get_group_and_liberties(index);
        if (our_liberties.size() == 1 && our_group.size() == 1) {
            ko_point_ = last_captured_idx;
        }
    }

    // Reset passes
    passes_ = 0;
    move_count_++;

    // Switch player
    to_play_ = opponent;

    return true;
}

bool GoBoard::pass() {
    passes_++;
    move_count_++;
    to_play_ = (to_play_ == BLACK) ? WHITE : BLACK;
    ko_point_ = std::nullopt;
    return true;
}

float GoBoard::score() const {
    // Chinese rules: area scoring
    float black_score = 0.0f;
    float white_score = komi_;

    // Count stones
    for (int i = 0; i < size_ * size_; ++i) {
        if (board_[i] == BLACK) {
            black_score += 1.0f;
        } else if (board_[i] == WHITE) {
            white_score += 1.0f;
        }
    }

    // Count territory (empty regions surrounded by one color)
    std::vector<bool> visited(size_ * size_, false);

    for (int i = 0; i < size_ * size_; ++i) {
        if (board_[i] != EMPTY || visited[i]) {
            continue;
        }

        // Flood-fill to find empty region
        std::vector<int> territory;
        std::unordered_set<int8_t> borders;
        std::queue<int> queue;
        queue.push(i);
        visited[i] = true;

        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            territory.push_back(current);

            for (int j = 0; j < neighbor_counts_[current]; ++j) {
                int neighbor = neighbor_indices_[current][j];
                if (board_[neighbor] == EMPTY) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.push(neighbor);
                    }
                } else {
                    borders.insert(board_[neighbor]);
                }
            }
        }

        // If territory is surrounded by only one color, count it
        if (borders.size() == 1) {
            float territory_size = static_cast<float>(territory.size());
            if (*borders.begin() == BLACK) {
                black_score += territory_size;
            } else {
                white_score += territory_size;
            }
        }
    }

    return black_score - white_score;
}

int8_t GoBoard::get_winner() const {
    float s = score();
    if (s > 0) {
        return BLACK;
    } else if (s < 0) {
        return WHITE;
    }
    return 0;  // Draw
}

void GoBoard::set_from_array(const int8_t* board_data, int8_t to_play) {
    // Copy board data
    for (int i = 0; i < size_ * size_; ++i) {
        board_[i] = board_data[i];
    }
    to_play_ = to_play;
    // Reset game state - ko detection would need move history
    ko_point_ = std::nullopt;
    passes_ = 0;
    // Count non-empty cells as rough move count estimate
    move_count_ = 0;
    for (int i = 0; i < size_ * size_; ++i) {
        if (board_[i] != EMPTY) {
            move_count_++;
        }
    }
}

}  // namespace alpha_go
