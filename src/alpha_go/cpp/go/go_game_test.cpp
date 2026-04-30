#include "go_game.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace alpha_go;

TEST_CASE("GoBoard construction", "[go_game]") {
    GoBoard board(9);

    REQUIRE(board.size() == 9);
    REQUIRE(board.to_play() == GoBoard::BLACK);
    REQUIRE(board.passes() == 0);
    REQUIRE(board.move_count() == 0);
    REQUIRE_FALSE(board.is_game_over());

    // All positions should be empty
    for (int i = 0; i < 81; ++i) {
        REQUIRE(board.at_flat(i) == GoBoard::EMPTY);
    }
}

TEST_CASE("Basic stone placement", "[go_game]") {
    GoBoard board(9);

    // Black plays at (4, 4)
    REQUIRE(board.play(4, 4));
    REQUIRE(board.at(4, 4) == GoBoard::BLACK);
    REQUIRE(board.to_play() == GoBoard::WHITE);
    REQUIRE(board.move_count() == 1);

    // White plays at (4, 5)
    REQUIRE(board.play(4, 5));
    REQUIRE(board.at(4, 5) == GoBoard::WHITE);
    REQUIRE(board.to_play() == GoBoard::BLACK);
    REQUIRE(board.move_count() == 2);

    // Cannot play on occupied position
    REQUIRE_FALSE(board.is_legal(4, 4));
    REQUIRE_FALSE(board.play(4, 4));
}

TEST_CASE("Single stone capture", "[go_game]") {
    GoBoard board(9);

    // Set up capture: Black surrounds White stone at corner
    // W at (0,0), Black at (0,1) and (1,0)
    board.play(0, 1);  // Black
    board.play(0, 0);  // White (to be captured)
    board.play(1, 0);  // Black (completes capture)

    // White stone should be captured
    REQUIRE(board.at(0, 0) == GoBoard::EMPTY);
}

TEST_CASE("Group capture", "[go_game]") {
    GoBoard board(9);

    // Build a white group and surround it
    // W W at (1,0) (1,1)
    // Surround with Black

    board.play(0, 0);  // Black
    board.play(1, 0);  // White
    board.play(0, 1);  // Black
    board.play(1, 1);  // White
    board.play(2, 0);  // Black
    board.play(8, 8);  // White (elsewhere)
    board.play(2, 1);  // Black
    board.pass();      // White
    board.play(1, 2);  // Black - completes capture

    // White group should be captured
    REQUIRE(board.at(1, 0) == GoBoard::EMPTY);
    REQUIRE(board.at(1, 1) == GoBoard::EMPTY);
}

TEST_CASE("Ko rule", "[go_game]") {
    GoBoard board(9);

    // Set up ko situation:
    //   0 1 2 3
    // 0 . B W .
    // 1 B . B W
    // 2 . B W .

    board.play(0, 1);  // Black
    board.play(0, 2);  // White
    board.play(1, 0);  // Black
    board.play(1, 3);  // White
    board.play(1, 2);  // Black
    board.play(2, 2);  // White
    board.play(2, 1);  // Black
    board.play(1, 1);  // White plays at (1,1), captures Black at (1,2)

    // Check white captured black
    REQUIRE(board.at(1, 2) == GoBoard::EMPTY);

    // Ko point should be set - Black cannot immediately recapture
    REQUIRE(board.ko_point().has_value());
    REQUIRE_FALSE(board.is_legal(1, 2));
}

TEST_CASE("Suicide is legal under Tromp-Taylor (self-capture)", "[go_game]") {
    GoBoard board(9);

    // Black surrounds (0,0). White's neighbors at (0,1) and (1,0) both have
    // outside liberties, so playing white at (0,0) doesn't capture them — it
    // just leaves the placed white stone with zero liberties. Under TT that's
    // a legal "suicide": the white stone is removed, the corner stays empty.
    board.play(0, 1);  // Black
    board.play(8, 8);  // White elsewhere
    board.play(1, 0);  // Black

    REQUIRE(board.is_legal(0, 0));
    REQUIRE(board.play(0, 0));      // White self-captures
    REQUIRE(board.at(0, 0) == GoBoard::EMPTY);
    REQUIRE(board.at(0, 1) == GoBoard::BLACK);
    REQUIRE(board.at(1, 0) == GoBoard::BLACK);
}

TEST_CASE("Capture is not suicide", "[go_game]") {
    GoBoard board(9);

    // Set up: White at corner (0,0), surrounded by Black except one liberty
    // If White plays and captures Black, it's not suicide

    //   0 1 2
    // 0 W B .
    // 1 B . .

    board.play(0, 1);  // Black
    board.play(0, 0);  // White (will be captured)
    board.play(1, 0);  // Black (captures)

    // White was captured
    REQUIRE(board.at(0, 0) == GoBoard::EMPTY);
}

TEST_CASE("Pass and game end", "[go_game]") {
    GoBoard board(9);

    REQUIRE(board.passes() == 0);
    REQUIRE_FALSE(board.is_game_over());

    board.pass();  // Black passes
    REQUIRE(board.passes() == 1);
    REQUIRE(board.to_play() == GoBoard::WHITE);
    REQUIRE_FALSE(board.is_game_over());

    board.pass();  // White passes
    REQUIRE(board.passes() == 2);
    REQUIRE(board.is_game_over());
}

TEST_CASE("Legal moves generation", "[go_game]") {
    GoBoard board(9);

    auto moves = board.get_legal_moves_flat();
    REQUIRE(moves.size() == 81);  // All positions legal at start

    board.play(4, 4);
    moves = board.get_legal_moves_flat();
    REQUIRE(moves.size() == 80);  // One less
}

TEST_CASE("Scoring empty board", "[go_game]") {
    GoBoard board(9);

    // Empty board: White wins by komi (7.5)
    float s = board.score();
    REQUIRE_THAT(s, Catch::Matchers::WithinAbs(-7.5f, 0.01f));
    REQUIRE(board.get_winner() == GoBoard::WHITE);
}

TEST_CASE("Scoring with territory", "[go_game]") {
    GoBoard board(9);

    // Play some stones to create territory
    // Black fills left column, White fills right column
    for (int row = 0; row < 9; ++row) {
        board.play(row, 0);  // Black on left
        board.play(row, 8);  // White on right
    }

    // Score should include stones and territory
    float s = board.score();
    // Both have 9 stones, middle territory is contested
    // With komi of 7.5, score should favor white slightly
}

TEST_CASE("Custom komi", "[go_game]") {
    GoBoard board(9, 5.5f);
    REQUIRE(board.komi() == 5.5f);

    // Empty board with 5.5 komi
    float s = board.score();
    REQUIRE_THAT(s, Catch::Matchers::WithinAbs(-5.5f, 0.01f));
}

TEST_CASE("Default komi", "[go_game]") {
    GoBoard board(9);
    REQUIRE(board.komi() == 7.5f);
}

TEST_CASE("Komi preserved on copy", "[go_game]") {
    GoBoard board(9, 5.5f);
    GoBoard copy(board);
    REQUIRE(copy.komi() == 5.5f);
}

TEST_CASE("Board copy", "[go_game]") {
    GoBoard board(9);
    board.play(4, 4);
    board.play(4, 5);

    GoBoard copy = board;

    REQUIRE(copy.at(4, 4) == GoBoard::BLACK);
    REQUIRE(copy.at(4, 5) == GoBoard::WHITE);
    REQUIRE(copy.to_play() == board.to_play());
    REQUIRE(copy.move_count() == board.move_count());

    // Modifying copy doesn't affect original
    copy.play(0, 0);
    REQUIRE(board.at(0, 0) == GoBoard::EMPTY);
    REQUIRE(copy.at(0, 0) == GoBoard::BLACK);
}
