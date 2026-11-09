#include <duckdb.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct Database {
    duckdb_database handle = nullptr;
    duckdb_connection connection = nullptr;

    ~Database() {
        duckdb_disconnect(&connection);
        duckdb_close(&handle);
    }

    void open(const char* filename) {
        duckdb_config config = nullptr;
        if (duckdb_create_config(&config) == DuckDBError) {
            throw std::runtime_error("Cannot create DuckDB configuration");
        }
        char* error = nullptr;
        auto status = duckdb_set_config(config, "autoinstall_known_extensions", "false");
        if (status == DuckDBSuccess && filename) {
            status = duckdb_set_config(config, "access_mode", "read_only");
        }
        if (status == DuckDBSuccess) {
            status = duckdb_open_ext(filename, &handle, config, &error);
        }
        duckdb_destroy_config(&config);
        std::unique_ptr<char, decltype(&duckdb_free)> message(error, &duckdb_free);
        if (status == DuckDBError) {
            throw std::runtime_error(message ? message.get() : "Cannot open DuckDB");
        }
        if (duckdb_connect(handle, &connection) == DuckDBError) {
            throw std::runtime_error("Cannot connect to DuckDB");
        }
    }
};

struct Statement {
    duckdb_prepared_statement handle = nullptr;
    ~Statement() { duckdb_destroy_prepare(&handle); }
};

struct Result {
    duckdb_result handle{};
    ~Result() { duckdb_destroy_result(&handle); }

    std::string text(idx_t column, idx_t row) {
        std::unique_ptr<char, decltype(&duckdb_free)> value(
            duckdb_value_varchar(&handle, column, row), &duckdb_free);
        return value ? std::string(value.get()) : std::string{};
    }
};

void query(Database& db, Result& result, const char* sql,
           std::initializer_list<const char*> arguments = {}) {
    Statement statement;
    if (duckdb_prepare(db.connection, sql, &statement.handle) == DuckDBError) {
        const char* message = duckdb_prepare_error(statement.handle);
        throw std::runtime_error(message ? message : "Cannot prepare retrieval query");
    }
    idx_t index = 1;
    for (const char* argument : arguments) {
        if (duckdb_bind_varchar(statement.handle, index++, argument) == DuckDBError) {
            throw std::runtime_error("Cannot bind retrieval query parameter");
        }
    }
    if (duckdb_execute_prepared(statement.handle, &result.handle) == DuckDBError) {
        const char* message = duckdb_result_error(&result.handle);
        throw std::runtime_error(message ? message : "Cannot execute retrieval query");
    }
}

std::uint32_t rotate_left(std::uint32_t value, int bits) {
    return (value << bits) | (value >> (32 - bits));
}

std::uint32_t mix_word(std::uint32_t word) {
    word *= 0xcc9e2d51U;
    word = rotate_left(word, 15);
    return word * 0x1b873593U;
}

// MurmurHash3 x86/32 with seed zero, matching the Python hashing vectorizer.
std::uint32_t hash_token(std::string_view token) {
    std::uint32_t hash = 0;
    for (std::size_t offset = 0; offset < token.size(); offset += 4) {
        const auto width = std::min(std::size_t{4}, token.size() - offset);
        std::uint32_t word = 0;
        for (std::size_t byte = 0; byte < width; ++byte) {
            word |= std::uint32_t(static_cast<unsigned char>(token[offset + byte])) << (8 * byte);
        }
        hash ^= mix_word(word);
        if (width == 4) {
            hash = rotate_left(hash, 13) * 5U + 0xe6546b64U;
        }
    }
    hash ^= static_cast<std::uint32_t>(token.size());
    hash ^= hash >> 16;
    hash *= 0x85ebca6bU;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35U;
    return hash ^ (hash >> 16);
}

using Vector = std::unordered_map<std::uint32_t, double>;

Vector vectorize(std::string_view terms) {
    Vector counts;
    while (!terms.empty()) {
        const auto end = terms.find(' ');
        const auto token = terms.substr(0, end);
        if (!token.empty()) {
            const auto hash = hash_token(token);
            const auto magnitude = hash & 0x80000000U ? 0U - hash : hash;
            ++counts[magnitude % (1U << 14)];
        }
        if (end == std::string_view::npos) {
            break;
        }
        terms.remove_prefix(end + 1);
    }
    return counts;
}

double cosine(const Vector& left, const Vector& right) {
    double dot = 0;
    double left_norm = 0;
    double right_norm = 0;
    for (const auto& [bucket, count] : left) {
        left_norm += count * count;
        const auto match = right.find(bucket);
        if (match != right.end()) {
            dot += count * match->second;
        }
    }
    for (const auto& entry : right) {
        right_norm += entry.second * entry.second;
    }
    return left_norm && right_norm ? dot / std::sqrt(left_norm * right_norm) : 0.0;
}

struct Report {
    std::string id;
    std::string document;
    double score;
};

constexpr const char* stats_sql = R"SQL(
    SELECT json_object(
        'source_id', 'stats:' || game_id || ':' || CAST(player_id AS VARCHAR),
        'source_url', source_url, 'published_at', CAST(published_at AS VARCHAR),
        'available_at', CAST(available_at AS VARCHAR), 'retrieved_at', CAST(retrieved_at AS VARCHAR),
        'timestamp_quality', timestamp_quality, 'kind', 'numerical',
        'text', 'Recorded points ' || points || '; minutes ' || minutes || ' in game ' || game_id || '.',
        'points', points, 'minutes', minutes)
    FROM logs
    WHERE CAST(player_id AS VARCHAR) = ?
      AND CAST(available_at AS TIMESTAMPTZ_NS) <= CAST(? AS TIMESTAMPTZ_NS)
      AND CAST(tipoff AS TIMESTAMPTZ_NS) < CAST(? AS TIMESTAMPTZ_NS)
      AND played = true AND minutes > 0
      AND season = ? AND CAST(team_id AS VARCHAR) = ? AND CAST(opponent_id AS VARCHAR) = ?
    ORDER BY tipoff DESC, game_id LIMIT 10
)SQL";

constexpr const char* query_terms_sql = R"SQL(
    SELECT array_to_string(regexp_extract_all(lower(?), '[\p{L}\p{N}_]{2,}'), ' ')
)SQL";

constexpr const char* reports_sql = R"SQL(
    SELECT CAST(source_id AS VARCHAR), row_to_json(report),
           array_to_string(regexp_extract_all(lower(CAST(text AS VARCHAR)), '[\p{L}\p{N}_]{2,}'), ' ')
    FROM read_parquet(?) AS report
    WHERE CAST(available_at AS TIMESTAMPTZ_NS) <= CAST(? AS TIMESTAMPTZ_NS)
      AND CAST(published_at AS TIMESTAMPTZ_NS) <= CAST(? AS TIMESTAMPTZ_NS)
      AND CAST(retrieved_at AS TIMESTAMPTZ_NS) <= CAST(? AS TIMESTAMPTZ_NS)
      AND CAST(season AS VARCHAR) = ?
      AND (CAST(player_id AS VARCHAR) = ? OR CAST(team_id AS VARCHAR) = ?)
      AND (opponent_id IS NULL OR CAST(opponent_id AS VARCHAR) = ?)
)SQL";

constexpr std::string_view usage =
    "Usage: nba-retrieve DATABASE REPORTS_PARQUET PLAYER TEAM OPPONENT SEASON CUTOFF QUERY TOP_K\n"
    "Use - for an omitted database or report file. CUTOFF is an ISO-8601 timestamp with a time zone.\n";

std::size_t parse_limit(std::string_view value) {
    std::size_t limit = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), limit);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("TOP_K must be a nonnegative integer");
    }
    return limit;
}

void retrieve(char** args) {
    const char* database = std::string_view(args[1]) == "-" ? nullptr : args[1];
    const char* cutoff = args[7];
    const auto limit = parse_limit(args[9]);

    Database db;
    db.open(database);
    Result utc_setting;
    query(db, utc_setting, "SET TimeZone='UTC'");
    Result checked_cutoff;
    query(db, checked_cutoff, "SELECT isfinite(CAST(? AS TIMESTAMPTZ))", {cutoff});
    if (!duckdb_value_boolean(&checked_cutoff.handle, 0, 0)) {
        throw std::invalid_argument("CUTOFF must be a finite timestamp");
    }

    Result stats;
    if (database) {
        query(db, stats, stats_sql, {args[3], cutoff, cutoff, args[6], args[4], args[5]});
    }

    std::vector<Report> ranked;
    if (std::string_view(args[2]) != "-" && limit > 0) {
        Result terms;
        query(db, terms, query_terms_sql, {args[8]});
        const auto search = vectorize(terms.text(0, 0));

        Result candidates;
        query(db, candidates, reports_sql,
              {args[2], cutoff, cutoff, cutoff, args[6], args[3], args[4], args[5]});
        for (idx_t row = 0; row < duckdb_row_count(&candidates.handle); ++row) {
            const auto score = cosine(search, vectorize(candidates.text(2, row)));
            ranked.push_back({candidates.text(0, row), candidates.text(1, row), score});
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const Report& left, const Report& right) {
            const auto left_score = std::round(left.score * 1e12);
            const auto right_score = std::round(right.score * 1e12);
            return left_score == right_score ? left.id < right.id : left_score > right_score;
        });
    }

    std::cout << "{\"numerical\":[";
    for (idx_t row = 0; database && row < duckdb_row_count(&stats.handle); ++row) {
        if (row) std::cout << ',';
        std::cout << stats.text(0, row);
    }
    std::cout << "],\"reports\":[" << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t row = 0; row < std::min(limit, ranked.size()); ++row) {
        if (row) std::cout << ',';
        std::cout << "{\"document\":" << ranked[row].document << ",\"score\":" << ranked[row].score << '}';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage;
        return 0;
    }
    if (argc != 10) {
        std::cerr << usage;
        return 2;
    }
    try {
        retrieve(argv);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nba-retrieve: " << error.what() << '\n';
        return 1;
    }
}
