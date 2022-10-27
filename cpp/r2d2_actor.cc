#include "rela/r2d2_actor.h"

torch::Tensor aggregatePriority(torch::Tensor priority, torch::Tensor seqLen, float eta) {
    assert(priority.device().is_cpu() && seqLen.device().is_cpu());
    auto mask = torch::arange(0, priority.size(0));
    mask = (mask.unsqueeze(1) < seqLen.unsqueeze(0)).to(torch::kFloat32);
    assert(priority.sizes() == mask.sizes());
    priority = priority * mask;

    auto pMean = priority.sum(0) / seqLen;
    auto pMax = std::get<0>(priority.max(0));
    auto aggPriority = eta * pMax + (1.0 - eta) * pMean;
    return aggPriority.detach();
}
