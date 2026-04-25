using System.IO;
using System.Collections.Concurrent;
using System.Text.Json;
using ChineseChessAI.Core;
using ChineseChessAI.Utils;

namespace ChineseChessAI.Traditional
{
    public enum OpeningBookMode
    {
        Off,
        Best,
        Weighted
    }

    public sealed class OpeningBook
    {
        private readonly Dictionary<ulong, Dictionary<Move, int>> _entries = new();
        private readonly MoveGenerator _generator = new();
        private static readonly ConcurrentDictionary<string, OpeningBook> SharedCaches = new();

        public int PositionCount => _entries.Count;

        public int MaxPly { get; }

        public OpeningBook(int maxPly = 24)
        {
            MaxPly = Math.Max(1, maxPly);
        }

        public static OpeningBook LoadDefaultCache(int maxPly = 24, string fileName = "opening_book.json")
        {
            string? cachePath = FindRepoFile(Path.Combine("data", fileName));
            if (cachePath == null)
                return new OpeningBook(maxPly);

            string cacheKey = $"{Path.GetFullPath(cachePath)}|{maxPly}";
            return SharedCaches.GetOrAdd(cacheKey, _ =>
            {
                var book = new OpeningBook(maxPly);
                book.LoadCache(cachePath);
                return book;
            });
        }

        public bool TryGetMove(Board board, OpeningBookMode mode, out Move move)
        {
            move = default;
            if (mode == OpeningBookMode.Off)
                return false;

            if (!_entries.TryGetValue(board.CurrentHash, out var moves) || moves.Count == 0)
                return false;

            var legalSet = _generator.GenerateLegalMoves(board, skipPerpetualCheck: false).ToHashSet();
            var candidates = moves
                .Where(kvp => legalSet.Contains(kvp.Key))
                .OrderByDescending(kvp => kvp.Value)
                .ToArray();

            if (candidates.Length == 0)
                return false;

            if (mode == OpeningBookMode.Best || candidates.Length == 1)
            {
                move = candidates[0].Key;
                return true;
            }

            int total = candidates.Sum(x => x.Value);
            int roll = Random.Shared.Next(total);
            int cumulative = 0;
            foreach (var candidate in candidates)
            {
                cumulative += candidate.Value;
                if (roll < cumulative)
                {
                    move = candidate.Key;
                    return true;
                }
            }

            move = candidates[0].Key;
            return true;
        }

        public int GetMoveFrequency(Board board, Move move)
        {
            if (!_entries.TryGetValue(board.CurrentHash, out var moves))
                return 0;

            return moves.TryGetValue(move, out int count) ? count : 0;
        }

        public int GetMoveOrderingBonus(Board board, Move move)
        {
            int count = GetMoveFrequency(board, move);
            if (count <= 0)
                return 0;

            return Math.Min(90_000, 5_000 + (int)(Math.Log2(count + 1) * 12_000));
        }

        public IReadOnlyList<(Move Move, int Count)> GetBookMoves(Board board, int limit = 16)
        {
            if (!_entries.TryGetValue(board.CurrentHash, out var moves) || moves.Count == 0)
                return Array.Empty<(Move, int)>();

            var legalSet = _generator.GenerateLegalMoves(board, skipPerpetualCheck: false).ToHashSet();
            return moves
                .Where(kvp => legalSet.Contains(kvp.Key))
                .OrderByDescending(kvp => kvp.Value)
                .Take(Math.Max(1, limit))
                .Select(kvp => (kvp.Key, kvp.Value))
                .ToArray();
        }

        public int LoadFromMasterDataDirectory(string directory, int maxGames = int.MaxValue)
        {
            if (!Directory.Exists(directory))
                return 0;

            int loaded = 0;
            foreach (string file in Directory.EnumerateFiles(directory, "*.json").Take(maxGames))
            {
                try
                {
                    string json = File.ReadAllText(file);
                    var game = JsonSerializer.Deserialize<MasterGameData>(json);
                    if (game?.MoveHistoryUcci == null || game.MoveHistoryUcci.Count == 0)
                        continue;

                    AddGame(game.MoveHistoryUcci);
                    loaded++;
                }
                catch
                {
                    // Ignore individual corrupt book source files.
                }
            }

            return loaded;
        }

        public int LoadFromUciPgnFile(string path, int maxGames = int.MaxValue)
        {
            if (!File.Exists(path))
                return 0;

            int loaded = 0;
            foreach (string line in File.ReadLines(path))
            {
                if (loaded >= maxGames)
                    break;

                var moves = ParseUciGameLine(line);
                if (moves.Count == 0)
                    continue;

                AddGame(moves);
                loaded++;
            }

            return loaded;
        }

        public int LoadFromPath(string path, int maxGames = int.MaxValue)
        {
            if (Directory.Exists(path))
                return LoadFromMasterDataDirectory(path, maxGames);

            string extension = Path.GetExtension(path).ToLowerInvariant();
            if (extension is ".pgn" or ".uci" or ".txt")
                return LoadFromUciPgnFile(path, maxGames);

            return 0;
        }

        public bool LoadCache(string path)
        {
            if (!File.Exists(path))
                return false;

            string json = File.ReadAllText(path);
            var cache = JsonSerializer.Deserialize<OpeningBookCache>(json);
            if (cache?.Positions == null)
                return false;

            _entries.Clear();
            foreach (var position in cache.Positions)
            {
                if (!ulong.TryParse(position.Hash, out ulong hash))
                    continue;

                var moves = new Dictionary<Move, int>();
                foreach (var moveEntry in position.Moves)
                {
                    Move? move = NotationConverter.UcciToMove(moveEntry.Move);
                    if (move.HasValue && moveEntry.Count > 0)
                        moves[move.Value] = moveEntry.Count;
                }

                if (moves.Count > 0)
                    _entries[hash] = moves;
            }

            return _entries.Count > 0;
        }

        public void SaveCache(string path)
        {
            string? directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            var cache = new OpeningBookCache
            {
                MaxPly = MaxPly,
                Positions = _entries
                    .Select(position => new OpeningBookPosition
                    {
                        Hash = position.Key.ToString(),
                        Moves = position.Value
                            .OrderByDescending(move => move.Value)
                            .Select(move => new OpeningBookMove
                            {
                                Move = NotationConverter.MoveToUcci(move.Key),
                                Count = move.Value
                            })
                            .ToList()
                    })
                    .ToList()
            };

            File.WriteAllText(path, JsonSerializer.Serialize(cache));
        }

        public void AddGame(IEnumerable<string> ucciMoves)
        {
            var board = new Board();
            int ply = 0;
            foreach (string ucci in ucciMoves)
            {
                if (ply >= MaxPly)
                    break;

                Move? parsed = NotationConverter.UcciToMove(ucci);
                if (!parsed.HasValue)
                    break;

                var move = parsed.Value;
                var legalMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: true);
                if (!legalMoves.Any(m => m.Equals(move)))
                    break;

                AddMove(board.CurrentHash, move);
                board.Push(move.From, move.To);
                ply++;
            }
        }

        public void Prune(int minCount, int topMovesPerPosition)
        {
            minCount = Math.Max(1, minCount);
            topMovesPerPosition = Math.Max(1, topMovesPerPosition);
            foreach (ulong hash in _entries.Keys.ToArray())
            {
                var keptMoves = _entries[hash]
                    .Where(kvp => kvp.Value >= minCount)
                    .OrderByDescending(kvp => kvp.Value)
                    .Take(topMovesPerPosition)
                    .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

                if (keptMoves.Count == 0)
                    _entries.Remove(hash);
                else
                    _entries[hash] = keptMoves;
            }
        }

        private void AddMove(ulong hash, Move move)
        {
            if (!_entries.TryGetValue(hash, out var moves))
            {
                moves = new Dictionary<Move, int>();
                _entries[hash] = moves;
            }

            moves.TryGetValue(move, out int count);
            moves[move] = count + 1;
        }

        private static List<string> ParseUciGameLine(string line)
        {
            var moves = new List<string>();
            foreach (string rawToken in line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                string token = rawToken.Trim();
                if (token.Length == 0 || token[0] == '[' || token.Contains('.'))
                    continue;

                if (token is "1-0" or "0-1" or "1/2-1/2" or "*" or "red" or "black" or "draw")
                    continue;

                if (IsUciMove(token))
                    moves.Add(token[..4]);
            }

            return moves;
        }

        private static bool IsUciMove(string token)
        {
            if (token.Length < 4)
                return false;

            return IsFile(token[0])
                && IsRank(token[1])
                && IsFile(token[2])
                && IsRank(token[3]);
        }

        private static bool IsFile(char c) => c is >= 'a' and <= 'i';

        private static bool IsRank(char c) => c is >= '0' and <= '9';

        private static string? FindRepoFile(string relativePath)
        {
            string directory = AppDomain.CurrentDomain.BaseDirectory;
            for (int i = 0; i < 10; i++)
            {
                string candidate = Path.Combine(directory, relativePath);
                if (File.Exists(candidate))
                    return candidate;

                string? parent = Directory.GetParent(directory)?.FullName;
                if (parent == null)
                    break;

                directory = parent;
            }

            string cwdCandidate = Path.Combine(Directory.GetCurrentDirectory(), relativePath);
            return File.Exists(cwdCandidate) ? cwdCandidate : null;
        }

        private sealed class OpeningBookCache
        {
            public int MaxPly { get; set; }
            public List<OpeningBookPosition> Positions { get; set; } = new();
        }

        private sealed class OpeningBookPosition
        {
            public string Hash { get; set; } = string.Empty;
            public List<OpeningBookMove> Moves { get; set; } = new();
        }

        private sealed class OpeningBookMove
        {
            public string Move { get; set; } = string.Empty;
            public int Count { get; set; }
        }
    }
}
