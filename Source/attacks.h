#ifndef ATTACKS_H
#define ATTACKS_H

#include "structs.h"
#include <stdint.h>
#include "immintrin.h"

extern const int bishop_relevant_bits[64];
extern const int rook_relevant_bits[64];
extern const uint64_t rook_magic_numbers[64];
extern const uint64_t bishop_magic_numbers[64];

extern uint64_t pawn_attacks[2][64];
extern uint64_t knight_attacks[64];
extern uint64_t king_attacks[64];
extern uint64_t bishop_attacks[64][512];
extern uint64_t rook_attacks[64][4096];
extern uint64_t bishop_masks[64];
extern uint64_t rook_masks[64];
extern uint64_t file_masks[64];
extern uint64_t rank_masks[64];
extern uint64_t isolated_masks[64];
extern uint64_t white_passed_masks[64];
extern uint64_t black_passed_masks[64];

int is_square_attacked(position_t *pos, int square, int side);
uint8_t stm_in_check(position_t *pos);
void init_sliders_attacks(void);
void init_leapers_attacks(void);

#if defined(USE_AVX2) || defined(USE_AVX512)
#define USE_MERLINS_ATTACKS
#endif

#ifdef USE_MERLINS_ATTACKS
typedef struct merlin_magic_s {
  uint64_t mask_file, mask_diag, mask_none, mask_antidiag;
  uint64_t r, rr;

  const uint8_t* restrict rank_attacks_lookup;
  int shift;
} merlin_magic_t;

extern merlin_magic_t merlins[64];

static inline __m256i bswap(__m256i x) {
  return _mm256_shuffle_epi8(x, _mm256_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7));
}

static inline void get_both_attacks(int square, uint64_t occupancy, uint64_t* bishop, uint64_t* rook) {
  const merlin_magic_t* merlin = &merlins[square];

  const __m256i mask = _mm256_load_si256((const __m256i*) merlin);
  const __m256i rs   = _mm256_set1_epi64x(merlin->r);
  const __m256i rrs  = _mm256_set1_epi64x(merlin->rr);

  __m256i o      = _mm256_and_si256(mask, _mm256_set1_epi64x(occupancy));
  __m256i fwd    = _mm256_sub_epi64(o, rs);
  __m256i rev    = bswap(_mm256_sub_epi64(bswap(o), rrs));
  __m256i result = _mm256_and_si256(_mm256_xor_si256(fwd, rev), mask);

  // Lane 0: rook attacks (file only); lane 1: bishop attacks
  __m128i rookBishop =
    _mm_or_si128(_mm256_extracti128_si256(result, 1), _mm256_castsi256_si128(result));

  uint64_t rowOccupancy = merlin->rank_attacks_lookup[occupancy >> (merlin->shift + 1) & 0x3f];
  uint64_t rankAttacks  = rowOccupancy << merlin->shift;

  // [bishop, rook]
  *bishop = _mm_extract_epi64(rookBishop, 1);
  *rook = _mm_cvtsi128_si64(rookBishop) + rankAttacks;
}
#endif

// get bishop attacks
static inline uint64_t get_bishop_attacks(int square, uint64_t occupancy) {
#ifdef USE_MERLINS_ATTACKS
  if (__builtin_constant_p(occupancy) && occupancy == 0) {
    return bishop_masks[square];
  }
  uint64_t bishop, rook;
  get_both_attacks(square, occupancy, &bishop, &rook);
  return bishop;
#else
  // get bishop attacks assuming current board occupancy
  occupancy &= bishop_masks[square];
  occupancy *= bishop_magic_numbers[square];
  occupancy >>= 64 - bishop_relevant_bits[square];

  // return bishop attacks
  return bishop_attacks[square][occupancy];
#endif
}

// get rook attacks
static inline uint64_t get_rook_attacks(int square, uint64_t occupancy) {
#ifdef USE_MERLINS_ATTACKS
  if (__builtin_constant_p(occupancy) && occupancy == 0) {
    return rook_masks[square];
  }
  uint64_t bishop, rook;
  get_both_attacks(square, occupancy, &bishop, &rook);
  return rook;
#else
  // get rook attacks assuming current board occupancy
  occupancy &= rook_masks[square];
  occupancy *= rook_magic_numbers[square];
  occupancy >>= 64 - rook_relevant_bits[square];

  // return rook attacks
  return rook_attacks[square][occupancy];
#endif
}

// get queen attacks
static inline uint64_t get_queen_attacks(int square, uint64_t occupancy) {
#ifdef USE_MERLINS_ATTACKS
  return get_bishop_attacks(square, occupancy) | get_rook_attacks(square, occupancy);
#else
  // init result attacks bitboard
  uint64_t queen_attacks = 0ULL;

  // init bishop occupancies
  uint64_t bishop_occupancy = occupancy;

  // init rook occupancies
  uint64_t rook_occupancy = occupancy;

  // get bishop attacks assuming current board occupancy
  bishop_occupancy &= bishop_masks[square];
  bishop_occupancy *= bishop_magic_numbers[square];
  bishop_occupancy >>= 64 - bishop_relevant_bits[square];

  // get bishop attacks
  queen_attacks = bishop_attacks[square][bishop_occupancy];

  // get rook attacks assuming current board occupancy
  rook_occupancy &= rook_masks[square];
  rook_occupancy *= rook_magic_numbers[square];
  rook_occupancy >>= 64 - rook_relevant_bits[square];

  // get rook attacks
  queen_attacks |= rook_attacks[square][rook_occupancy];

  // return queen attacks
  return queen_attacks;
#endif
}

static inline uint64_t get_pawn_attacks(uint8_t side, int square) {
  return pawn_attacks[side][square];
}

static inline uint64_t get_knight_attacks(int square) {
  return knight_attacks[square];
}

static inline uint64_t get_king_attacks(int square) {
  return king_attacks[square];
}

uint64_t attackers_to(position_t *pos, int square, uint64_t occupancy);

void calculate_threats(position_t *pos, searchstack_t *ss);

uint8_t is_square_threatened(searchstack_t *ss, int square);

#endif
