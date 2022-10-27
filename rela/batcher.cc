#include "rela/batcher.h"

namespace rela {

TensorDict allocateBatchStorage(const TensorDict& data, int size) {
    TensorDict storage;
    for (const auto& kv : data) {
        auto t = kv.second.sizes();
        std::vector<int64_t> sizes;
        // for (int i = 0; i < batchdim_; ++i) {
        //   sizes.push_back(t[i]);
        // }
        sizes.push_back(size);
        for (size_t i = 0; i < t.size(); ++i) {
            sizes.push_back(t[i]);
        }

        storage[kv.first] = torch::zeros(sizes, kv.second.dtype());
    }
    return storage;
}

}

