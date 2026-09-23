#include "weight_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace fungt::mesher {

namespace {

class JsonParser {
private:
    const std::string& m_json;
    std::size_t m_pos;

    void skip_whitespace() {
        while (m_pos < m_json.size() && std::isspace(m_json[m_pos])) {
            ++m_pos;
        }
    }

    char peek() const {
        return m_pos < m_json.size() ? m_json[m_pos] : '\0';
    }

    char consume() {
        return m_pos < m_json.size() ? m_json[m_pos++] : '\0';
    }

    void expect(char c) {
        skip_whitespace();
        if (consume() != c) {
            throw std::runtime_error(
                std::string("JSON parse error: expected '") + c + "'");
        }
    }

    std::string parse_string() {
        skip_whitespace();
        if (consume() != '"') {
            throw std::runtime_error("JSON parse error: expected string");
        }
        std::string result;
        while (m_pos < m_json.size() && m_json[m_pos] != '"') {
            if (m_json[m_pos] == '\\' && m_pos + 1 < m_json.size()) {
                ++m_pos;
            }
            result += m_json[m_pos++];
        }
        ++m_pos;
        return result;
    }

    int parse_int() {
        skip_whitespace();
        std::size_t start = m_pos;
        if (m_json[m_pos] == '-') ++m_pos;
        while (m_pos < m_json.size() && std::isdigit(m_json[m_pos])) {
            ++m_pos;
        }
        return std::stoi(m_json.substr(start, m_pos - start));
    }

    std::vector<int> parse_int_array() {
        std::vector<int> result;
        skip_whitespace();
        expect('[');
        skip_whitespace();
        if (peek() != ']') {
            result.push_back(parse_int());
            skip_whitespace();
            while (peek() == ',') {
                consume();
                result.push_back(parse_int());
                skip_whitespace();
            }
        }
        expect(']');
        return result;
    }

public:
    struct WeightInfo {
        std::string name;
        std::string file;
        std::vector<int> shape;
        std::string dtype;
    };

    explicit JsonParser(const std::string& json) : m_json(json), m_pos(0) {}

    std::vector<WeightInfo> parse() {
        std::vector<WeightInfo> weights;

        skip_whitespace();
        expect('{');

        skip_whitespace();
        std::string key = parse_string();
        if (key != "weights") {
            throw std::runtime_error("JSON parse error: expected 'weights' key");
        }

        skip_whitespace();
        expect(':');
        skip_whitespace();
        expect('[');

        skip_whitespace();
        if (peek() != ']') {
            weights.push_back(parse_weight_entry());
            skip_whitespace();
            while (peek() == ',') {
                consume();
                weights.push_back(parse_weight_entry());
                skip_whitespace();
            }
        }

        expect(']');
        skip_whitespace();
        expect('}');

        return weights;
    }

    WeightInfo parse_weight_entry() {
        WeightInfo info;

        skip_whitespace();
        expect('{');

        for (int i = 0; i < 4; ++i) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            expect(':');

            if (key == "name") {
                info.name = parse_string();
            } else if (key == "file") {
                info.file = parse_string();
            } else if (key == "shape") {
                info.shape = parse_int_array();
            } else if (key == "dtype") {
                info.dtype = parse_string();
            }

            skip_whitespace();
            if (peek() == ',') {
                consume();
            }
        }

        skip_whitespace();
        expect('}');

        return info;
    }
};

}

ScopedWeightLoader::ScopedWeightLoader(WeightLoader& loader, const std::string& prefix)
    : m_loader(loader)
    , m_prefix(prefix.empty() ? "" : (prefix + "/"))
{
}

const std::vector<float>& ScopedWeightLoader::get(const std::string& name) {
    return m_loader.get(m_prefix + name);
}

std::vector<int> ScopedWeightLoader::shape(const std::string& name) const {
    return m_loader.shape(m_prefix + name);
}

WeightLoader::WeightLoader(const std::string& dir_path)
    : m_base_dir(dir_path)
{
    if (m_base_dir.empty()) {
        throw std::runtime_error("WeightLoader: directory path cannot be empty");
    }
    if (m_base_dir.back() != '/') {
        m_base_dir += '/';
    }
    parse_manifest(m_base_dir + "weights.json");
}

void WeightLoader::parse_manifest(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error(
            "WeightLoader: cannot open manifest file: " + json_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();

    JsonParser parser(json_content);
    auto weights = parser.parse();

    for (const auto& w : weights) {
        WeightEntry entry;
        entry.file = w.file;
        entry.shape = w.shape;
        entry.dtype = w.dtype;
        entry.loaded = false;
        m_manifest[w.name] = std::move(entry);
    }
}

std::size_t WeightLoader::compute_element_count(const std::vector<int>& shape) const {
    if (shape.empty()) return 0;
    std::size_t count = 1;
    for (int dim : shape) {
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

void WeightLoader::load_weight(const std::string& name) {
    auto it = m_manifest.find(name);
    if (it == m_manifest.end()) {
        throw std::runtime_error("WeightLoader: weight not found: " + name);
    }

    WeightEntry& entry = it->second;
    if (entry.loaded) return;

    std::string file_path = m_base_dir + entry.file;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(
            "WeightLoader: cannot open weight file: " + file_path);
    }

    std::size_t element_count = compute_element_count(entry.shape);
    entry.data.resize(element_count);

    file.read(reinterpret_cast<char*>(entry.data.data()),
              element_count * sizeof(float));

    if (!file) {
        throw std::runtime_error(
            "WeightLoader: failed to read weight file: " + file_path);
    }

    entry.loaded = true;
}

const std::vector<float>& WeightLoader::get(const std::string& name) {
    auto it = m_manifest.find(name);
    if (it == m_manifest.end()) {
        throw std::runtime_error("WeightLoader: weight not found: " + name);
    }

    if (!it->second.loaded) {
        load_weight(name);
    }

    return it->second.data;
}

std::vector<int> WeightLoader::shape(const std::string& name) const {
    auto it = m_manifest.find(name);
    if (it == m_manifest.end()) {
        throw std::runtime_error("WeightLoader: weight not found: " + name);
    }
    return it->second.shape;
}

ScopedWeightLoader WeightLoader::scope(const std::string& prefix) {
    return ScopedWeightLoader(*this, prefix);
}

bool WeightLoader::has(const std::string& name) const {
    return m_manifest.find(name) != m_manifest.end();
}

}
