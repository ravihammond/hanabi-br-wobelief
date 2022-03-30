#include <chrono>
#include <future>

#include "rlcc/sparta.h"

using namespace approx_search;

namespace sparta {

float searchMove(
    const hle::HanabiState& state,
    hle::HanabiMove move,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int> seeds,
    int myIdx,
    const std::vector<HybridModel>& players) {

  std::vector<std::vector<HybridModel>> allPlayers(hands.size(), players);
  std::vector<GameSimulator> games;
  games.reserve(hands.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    std::vector<SimHand> simHands{
        SimHand(myIdx, hands[i]),
    };
    games.emplace_back(state, simHands, seeds[i]);
  }

  size_t terminated = 0;
  std::vector<int> notTerminated;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(!games[i].terminal());
    games[i].step(move);
    if (!games[i].terminal()) {
      notTerminated.push_back(i);
    } else {
      ++terminated;
    }
  }

  while (!notTerminated.empty()) {
    std::vector<int> newNotTerminated;
    for (auto i : notTerminated) {
      assert(!games[i].state().IsTerminal());
      for (auto& actor : allPlayers[i]) {
        actor.observeBeforeAct(games[i], 0);
      }
    }

    for (auto i : notTerminated) {
      auto& game = games[i];
      int action = -1;
      for (auto& actor : allPlayers[i]) {
        int a = actor.decideAction(game, false);
        if (actor.index == game.state().CurPlayer()) {
          action = a;
        }
      }
      // std::cout << "move: " << game.getMove(action).ToString() << std::endl;
      game.step(game.getMove(action));
      if (!game.terminal()) {
        newNotTerminated.push_back(i);
      } else {
        ++terminated;
      }
    }

    notTerminated = newNotTerminated;
  }
  assert(terminated == games.size());

  std::vector<float> scores(games.size());
  float mean = 0;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(games[i].terminal());
    scores[i] = games[i].score();
    mean += scores[i];
  }
  mean = mean / scores.size();

  return mean;
}

// should be called after decideAction?
hle::HanabiMove SpartaActor::spartaSearch(
    const GameSimulator& env, hle::HanabiMove bpMove, int numSearch, float threshold, std::vector<int> cardCount, 
    std::vector<int> samples1,
    std::vector<int> samples2,
    std::vector<int> samples3,
    std::vector<int> samples4,
    std::vector<int> samples5) {
  torch::NoGradGuard ng;

  const auto& state = env.state();
  assert(state.CurPlayer() == index);
  auto legalMoves = state.LegalMoves(index);

  int numSearchPerMove = numSearch;// / legalMoves.size();
  std::cout << "SPARTA will run " << numSearchPerMove << " searches on "
            << legalMoves.size() << " legal moves" << std::endl;

//  auto simHands = handDist_.sampleHands(numSearchPerMove, &rng_);
  std::vector<std::vector<hle::HanabiCardValue>> simHands;
  for (int j = 0; j < numSearchPerMove; ++j) {
    bool validSample = true;
    auto cardRemain = cardCount;
    std::vector<hle::HanabiCardValue> hand;
    if (cardRemain[samples1[j]] > 0 ) {
	if (samples1[j] < 25) {
    		hand.emplace_back((int)samples1[j] / 5, samples1[j] % 5);
		--cardRemain[samples1[j]];
	}
    }
    else {
	validSample = false;
    }
    if (validSample && cardRemain[samples2[j]] > 0 ) {
	if (samples2[j] < 25) {
		 hand.emplace_back((int)samples2[j] / 5, samples2[j] % 5);
		 --cardRemain[samples2[j]];
	}
    }
    else {
        validSample = false;
    }
    if (validSample && cardRemain[samples3[j]] > 0 ) {
	if (samples3[j] < 25) {
	       	hand.emplace_back((int)samples3[j] / 5, samples3[j] % 5);
	       	--cardRemain[samples3[j]];	
	}
    }
    else {
        validSample = false;
    }
    if (validSample && cardRemain[samples4[j]] > 0 ) {
        if (samples4[j] < 25) { 
                hand.emplace_back((int)samples4[j] / 5, samples4[j] % 5); 
                --cardRemain[samples4[j]];
        }
    }
    else {
        validSample = false;
    }
    if (validSample && cardRemain[samples5[j]] > 0 ) {
        if (samples5[j] < 25) { 
                hand.emplace_back((int)samples5[j] / 5, samples5[j] % 5); 
                --cardRemain[samples5[j]];
        }
    }
    else {
        validSample = false;
    }
    if (validSample && state.Hands()[state.CurPlayer()].Cards().size() == hand.size() && state.Hands()[state.CurPlayer()].CanSetCards(hand)) {
   	simHands.push_back(hand);
    }
  }

  std::vector<int> seeds;
  for (size_t i = 0; i < simHands.size(); ++i) {
    seeds.push_back(int(rng_()));
  }

  std::vector<std::future<float>> futMoveScores;
  std::vector<HybridModel> players;
  for (auto& p : partners_) {
    players.push_back(p->model_);
  }
  for (auto move : legalMoves) {
    futMoveScores.push_back(std::async(
        std::launch::async, searchMove, state, move, simHands, seeds, index, players));
  }

  hle::HanabiMove bestMove = bpMove;
  float bpScore = -1;
  float bestScore = -1;

  std::cout << "SPARTA scores for moves:" << std::endl;
  for (size_t i = 0; i < legalMoves.size(); ++i) {
    float score = futMoveScores[i].get();
    auto move = legalMoves[i];
    if (move == bpMove) {
      assert(bpScore == -1);
      bpScore = score;
    }
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    std::cout << move.ToString() << ": " << score << std::endl;
  }

  std::cout << "SPARTA best - bp: " << bestScore - bpScore << std::endl;
  if (bestScore - bpScore >= threshold) {
    std::cout << "SPARTA changes move from " << bpMove.ToString() << " to "
              << bestMove.ToString() << std::endl;
    return bestMove;
  } else {
    return bpMove;
  }
}

}  // namespace sparta
