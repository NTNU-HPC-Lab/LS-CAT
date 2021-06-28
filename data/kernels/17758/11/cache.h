#ifndef TORCH_RADON_CACHE_H
#define TORCH_RADON_CACHE_H

#include <iostream>
#include <string.h>

using namespace std;

class DeviceSizeKey {
public:
    int device;

    int batch;
    int width;
    int height;

    int channels;
    int precision;

    DeviceSizeKey(int d, int b, int w, int h, int c=1, int p=1);

    bool operator==(const DeviceSizeKey &o) const;
};

std::ostream &operator<<(std::ostream &os, DeviceSizeKey const &m);

template<typename Key, typename Value>
class Cache {
    Value **cache;
    size_t cache_size;
public:
    Cache(size_t cache_size) {
        this->cache_size = cache_size;
        size_t size = cache_size * sizeof(Value *);
        this->cache = (Value **) malloc(size);
        memset(this->cache, 0, size);
    }

    bool exists(Key k) {
        for (uint i = 0; i < this->cache_size; i++) {
            if (this->cache[i] == 0) return false;
            if (this->cache[i]->matches(k)) return true;
        }
        return false;
    }

    Value *get(Key k) {
        uint i;
        for (i = 0; i < this->cache_size; i++) {
            if (this->cache[i] == 0 || this->cache[i]->matches(k)) break;
        }

        // cache is full and didn't match
        if (i == this->cache_size) {
            i -= 1;
            delete this->cache[i];
            this->cache[i] = 0;
        }

        // if needed allocate else move item closer to beginning of the cache
        if (this->cache[i] == 0) {
            this->cache[i] = new Value(k);
        } else {
            if (i > 0) {
                swap(this->cache[i - 1], this->cache[i]);
                i -= 1;
            }
        }

        return this->cache[i];
    }

    void free() {
#ifdef VERBOSE
        cout << "[TORCH RADON] Freeing cache" << endl;
#endif
        for (uint i = 0; i < this->cache_size; i++) {
            if (this->cache[i] != 0) {
                delete this->cache[i];
                this->cache[i] = 0;
            }
        }
    }

    ~Cache() {
        this->free();
        ::free((void *) this->cache);
    }
};


#endif //TORCH_RADON_CACHE_H
