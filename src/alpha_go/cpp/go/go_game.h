#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace alpha_go {

// Board representation: flat array for cache efficiency
// Values: 0=EMPTY, 1=BLACK, 2=WHITE
class GoBoard {
public:
    static constexpr int8_t EMPTY = 0;
    static constexpr int8_t BLACK = 1;
    static constexpr int8_t WHITE = 2;
    // 7.5 is recommended for high level / AI (black's first-move advantage is maximized)
    // 5.5 is recommended for casual games (to prevent game from being too hard for black)
    static constexpr float KOMI = 7.5f;

    explicit GoBoard(int size = 9, float komi = KOMI);
    GoBoard(const GoBoard&) = default;
    GoBoard& operator=(const GoBoard&) = default;
    GoBoard(GoBoard&&) = default;
    GoBoard& operator=(GoBoard&&) = default;

    // Core operations
    bool play(int row, int col);      // Returns false if illegal
    bool play_flat(int index);        // Flat index version
    bool pass();                      // Pass move

    // Queries
    bool is_legal(int row, int col) const;
    bool is_legal_flat(int index) const;
    std::vector<int> get_legal_moves_flat() const;  // Returns flat indices
    bool is_game_over() const { return passes_ >= 2; }
    float score() const;              // Tromp-Taylor area scoring, returns black - white (with komi)
    int8_t get_winner() const;        // BLACK, WHITE, or 0 for draw

    // State access
    int size() const { return size_; }
    int8_t at(int row, int col) const { return board_[row * size_ + col]; }
    int8_t at_flat(int index) const { return board_[index]; }
    int8_t to_play() const { return to_play_; }
    const int8_t* data() const { return board_.data(); }
    int passes() const { return passes_; }
    int move_count() const { return move_count_; }
    float komi() const { return komi_; }
    std::optional<int> ko_point() const { return ko_point_; }

    // Utility
    int flat_index(int row, int col) const { return row * size_ + col; }
    std::pair<int, int> row_col(int flat) const { return {flat / size_, flat % size_}; }

    // Set board state from array (for Python interop)
    // board_data: flat array of size*size int8_t values (EMPTY/BLACK/WHITE)
    // to_play: current player (BLACK or WHITE)
    void set_from_array(const int8_t* board_data, int8_t to_play);

private:
    void init_neighbors();

    // Flood-fill to find connected group and its liberties
    // Returns (group_indices, liberty_indices)
    std::pair<std::vector<int>, std::vector<int>> get_group_and_liberties(int index) const;

    // Remove all stones in a group
    int remove_group(const std::vector<int>& group);

    int size_;
    float komi_;
    std::vector<int8_t> board_;       // size_ * size_
    int8_t to_play_ = BLACK;
    std::optional<int> ko_point_;     // Flat index of ko point
    int passes_ = 0;
    int move_count_ = 0;

    // Pre-computed neighbors (4 neighbors per cell max)
    // neighbor_indices_[i] contains the flat indices of neighbors of cell i
    // neighbor_counts_[i] is the number of valid neighbors (1-4)
    std::vector<std::array<int, 4>> neighbor_indices_;
    std::vector<int> neighbor_counts_;
};

}  // namespace alpha_go
