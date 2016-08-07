#include <vector>
#include <map>
#include <sstream>
#include <string>

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>


class layer {
public:
    virtual std::string name() const = 0;
    virtual std::shared_ptr<layer> clone() const = 0;
};

#define IMPLEMENT_CLONE_AND_SERIALIZATION(type, layer_name) \
virtual std::string name() const {                          \
    return layer_name;                                      \
}                                                           \
virtual std::shared_ptr<layer> clone() const {              \
    return std::make_shared<type>(p);                       \
}                                                           \
template <class Archive>                                    \
void serialize(Archive & archive) {                         \
    p.serialize(archive);                                   \
}



//////////////////////////
// conv-layer

struct conv_param {
    size_t window_w;
    size_t window_h;
    size_t out_channels;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(CEREAL_NVP(window_w), CEREAL_NVP(window_h), CEREAL_NVP(out_channels));
    }
};


class conv_layer : public layer {
public:
    conv_layer() {}
    explicit conv_layer(const conv_param& p) : p(p) {}
    conv_layer(size_t window_size, size_t out_channels) : p({ window_size, window_size, out_channels }) {}

    IMPLEMENT_CLONE_AND_SERIALIZATION(conv_layer, "conv")

private:
    conv_param p;
};

//////////////////////////
// fc-layer

struct fc_param {
    size_t out_size;
    bool has_bias;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(CEREAL_NVP(out_size), CEREAL_NVP(has_bias));
    }
};

class fc_layer : public layer {
public:
    fc_layer() {}
    explicit fc_layer(const fc_param& p) : p(p) {}
    explicit fc_layer(size_t out_size) : p({out_size, true}) {}

    IMPLEMENT_CLONE_AND_SERIALIZATION(fc_layer, "fc")

private:
    fc_param p;
};

/////////////////////////////
// nodes(graph)

typedef layer* layerptr_t;

class nodes {
public:

    void add(std::shared_ptr<layer> layer) {
        own_nodes_.push_back(layer);
        nodes_.push_back(&*layer);
    }

    void add(layerptr_t layer) {
        nodes_.push_back(layer);       
    }


    void save(std::ostream& os) const {
        std::vector<std::shared_ptr<layer>> tmp(own_nodes_);
        // clone all layers which nodes don't have its ownership
        for (auto n : nodes_) {
            if (!own(n)) {
                tmp.push_back(n->clone());
            }
        }
        cereal::JSONOutputArchive o_archive(os);
        o_archive(tmp);

        // TODO: save graph-connection structure
    }

    void load(std::istream& is) {
        cereal::JSONInputArchive i_archive(is);
        i_archive(own_nodes_);

        for (auto& n : own_nodes_) {
            nodes_.push_back(&*n);
        }
    }
private:
    bool own(layerptr_t l) const {
        for (auto& n : own_nodes_) {
            if (&*n == l) return true;
        }
        return false;
    }
    std::vector<std::shared_ptr<layer> > own_nodes_;
    std::vector<layerptr_t> nodes_;
};

// Register DerivedClassOne
CEREAL_REGISTER_TYPE(conv_layer);

// Register EmbarassingDerivedClass with a less embarrasing name
CEREAL_REGISTER_TYPE(fc_layer);

// Note that there is no need to register the base class, only derived classes
//  However, since we did not use cereal::base_class, we need to clarify
//  the relationship (more on this later)
CEREAL_REGISTER_POLYMORPHIC_RELATION(layer, conv_layer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(layer, fc_layer);

int main(void) {
    fc_layer l1(3);
    conv_layer l2(5, 2);

    nodes n;
    n.add(std::make_shared<fc_layer>(l1));
    n.add(&l2);

    std::stringstream ss;
    n.save(ss);

    std::cout << ss.str();

    nodes n2;
    n2.load(ss);

    /*{
        cereal::JSONOutputArchive oarchive(std::cout);

        // Create instances of the derived classes, but only keep base class pointers
        std::shared_ptr<layer> ptr1 = std::make_shared<conv_layer>(cv);
        std::shared_ptr<layer> ptr2 = std::make_shared<fc_layer>(fc);
        oarchive(ptr1, ptr2);
    }*/
}