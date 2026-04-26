using System.Collections.Concurrent;
using System.IO;
using System.Text.Json;
using ChineseChessAI.Core;
using ChineseChessAI.Utils;

namespace ChineseChessAI.Traditional
{
    public sealed class MasterKnowledgeBook
    {
        private readonly Dictionary<ulong, Dictionary<Move, MasterMoveKnowledge>> _entries = new();
        private readonly MoveGenerator _generator = new();
        private static readonly ConcurrentDictionary<string, MasterKnowledgeBook> SharedCaches = new();

        public int PositionCount => _entries.Count;
        public int MaxPly { get; }
        public int LoadedGames { get; private set; }
        public int RedWinGames { get; private set; }
        public int BlackWinGames { get; private set; }
        public int DrawGames { get; private set; }
        public int UnknownGames { get; private set; }

        public MasterKnowledgeBook(int maxPly = 120)
        {
            MaxPly = Math.Max(1, maxPly);
        }

        public static MasterKnowledgeBook LoadDefaultCache(int maxPly = 120, string fileName = "master_knowledge_book.json")
        {
            string? cachePath = FindRepoFile(Path.Combine("data", fileName));
            if (cachePath == null)
                return new MasterKnowledgeBook(maxPly);

            string cacheKey = $"{Path.GetFullPath(cachePath)}|{maxPly}";
            return SharedCaches.GetOrAdd(cacheKey, _ =>
            {
                var book = new MasterKnowledgeBook(maxPly);
                book.LoadCache(cachePath);
                return book;
            });
        }

        public int GetMoveOrderingBonus(Board board, Move move)
        {
            if (!_entries.TryGetValue(board.CurrentHash, out var moves))
                return 0;
            if (!moves.TryGetValue(move, out var knowledge))
                return 0;

            int countBonus = Math.Min(120_000, 8_000 + (int)(Math.Log2(knowledge.Count + 1) * 14_000));
            int confidence = Math.Min(knowledge.Count, 40);
            int scoreBonus = knowledge.ScoreFromSideToMove * confidence / 40;
            return countBonus + scoreBonus;
        }

        public IReadOnlyList<MasterMoveKnowledge> GetMoves(Board board, int limit = 16)
        {
            if (!_entries.TryGetValue(board.CurrentHash, out var moves))
                return Array.Empty<MasterMoveKnowledge>();

            var legalSet = _generator.GenerateLegalMoves(board, skipPerpetualCheck: false).ToHashSet();
            return moves
                .Where(kvp => legalSet.Contains(kvp.Key))
                .OrderByDescending(kvp => kvp.Value.Count)
                .Take(Math.Max(1, limit))
                .Select(kvp => kvp.Value)
                .ToArray();
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

                    AddGame(game.MoveHistoryUcci, ParseResult(game.Result));
                    loaded++;
                }
                catch
                {
                }
            }

            return loaded;
        }

        public int LoadFromUciPgnFile(string path, int maxGames = int.MaxValue)
        {
            if (!File.Exists(path))
                return 0;

            int loaded = 0;
            var tags = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var moveText = new List<string>();

            foreach (string rawLine in File.ReadLines(path))
            {
                if (loaded >= maxGames)
                    break;

                string line = rawLine.Trim();
                if (line.Length == 0)
                {
                    loaded += FlushPgnGame(tags, moveText);
                    tags.Clear();
                    moveText.Clear();
                    continue;
                }

                if (TryParseTag(line, out string tagName, out string tagValue))
                {
                    tags[tagName] = tagValue;
                }
                else
                {
                    moveText.Add(line);
                }
            }

            if (loaded < maxGames)
                loaded += FlushPgnGame(tags, moveText);

            return Math.Min(loaded, maxGames);
        }

        public void Prune(int minCount, int topMovesPerPosition)
        {
            minCount = Math.Max(1, minCount);
            topMovesPerPosition = Math.Max(1, topMovesPerPosition);
            foreach (ulong hash in _entries.Keys.ToArray())
            {
                var keptMoves = _entries[hash]
                    .Where(kvp => kvp.Value.Count >= minCount)
                    .OrderByDescending(kvp => kvp.Value.Count)
                    .ThenByDescending(kvp => kvp.Value.ScoreFromSideToMove)
                    .Take(topMovesPerPosition)
                    .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

                if (keptMoves.Count == 0)
                    _entries.Remove(hash);
                else
                    _entries[hash] = keptMoves;
            }
        }

        public bool LoadCache(string path)
        {
            if (!File.Exists(path))
                return false;

            string json = File.ReadAllText(path);
            var cache = JsonSerializer.Deserialize<MasterKnowledgeCache>(json);
            if (cache?.Positions == null)
                return false;

            _entries.Clear();
            foreach (var position in cache.Positions)
            {
                if (!ulong.TryParse(position.Hash, out ulong hash))
                    continue;

                var moves = new Dictionary<Move, MasterMoveKnowledge>();
                foreach (var entry in position.Moves)
                {
                    Move? move = NotationConverter.UcciToMove(entry.Move);
                    if (!move.HasValue || entry.Count <= 0)
                        continue;

                    moves[move.Value] = new MasterMoveKnowledge(
                        move.Value,
                        entry.Count,
                        entry.RedWins,
                        entry.BlackWins,
                        entry.Draws,
                        entry.Unknown,
                        entry.SideToMove);
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

            var cache = new MasterKnowledgeCache
            {
                MaxPly = MaxPly,
                Positions = _entries
                    .Select(position => new MasterKnowledgePosition
                    {
                        Hash = position.Key.ToString(),
                        Moves = position.Value
                            .OrderByDescending(move => move.Value.Count)
                            .Select(move => new MasterKnowledgeMove
                            {
                                Move = NotationConverter.MoveToUcci(move.Key),
                                Count = move.Value.Count,
                                RedWins = move.Value.RedWins,
                                BlackWins = move.Value.BlackWins,
                                Draws = move.Value.Draws,
                                Unknown = move.Value.Unknown,
                                SideToMove = move.Value.SideToMove
                            })
                            .ToList()
                    })
                    .ToList()
            };

            File.WriteAllText(path, JsonSerializer.Serialize(cache));
        }

        private int FlushPgnGame(Dictionary<string, string> tags, List<string> moveText)
        {
            if (moveText.Count == 0)
                return 0;

            string resultText = tags.TryGetValue("Result", out string? value) ? value : string.Empty;
            var result = ParseResult(resultText);
            var moves = ParseUciMoves(string.Join(' ', moveText));
            if (moves.Count == 0)
                return 0;

            if (result == GameOutcome.Unknown)
                result = InferOutcomeFromFinalPosition(moves);

            AddGame(moves, result);
            return 1;
        }

        private void AddGame(IEnumerable<string> ucciMoves, GameOutcome result)
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

                AddMove(board.CurrentHash, move, board.IsRedTurn, result);
                board.Push(move.From, move.To);
                ply++;
            }

            AddGameResult(result);
        }

        private GameOutcome InferOutcomeFromFinalPosition(IReadOnlyList<string> ucciMoves)
        {
            var board = new Board();
            int noProgressPly = 0;

            foreach (string ucci in ucciMoves)
            {
                Move? parsed = NotationConverter.UcciToMove(ucci);
                if (!parsed.HasValue)
                    return GameOutcome.Unknown;

                var move = parsed.Value;
                var legalMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: true);
                if (!legalMoves.Any(m => m.Equals(move)))
                    return GameOutcome.Unknown;

                board.Push(move.From, move.To);
                noProgressPly = board.LastMoveWasIrreversible ? 0 : noProgressPly + 1;

                var kingOutcome = GetMissingKingOutcome(board);
                if (kingOutcome != GameOutcome.Unknown)
                    return kingOutcome;

                if (board.GetRepetitionCount() >= 3)
                    return GameOutcome.Draw;
                if (noProgressPly >= 100)
                    return GameOutcome.Draw;
            }

            var missingKingOutcome = GetMissingKingOutcome(board);
            if (missingKingOutcome != GameOutcome.Unknown)
                return missingKingOutcome;

            var finalLegalMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: true);
            if (finalLegalMoves.Count == 0)
                return board.IsRedTurn ? GameOutcome.BlackWin : GameOutcome.RedWin;

            return GameOutcome.Unknown;
        }

        private static GameOutcome GetMissingKingOutcome(Board board)
        {
            bool redKing = false;
            bool blackKing = false;
            for (int i = 0; i < 90; i++)
            {
                sbyte piece = board.GetPiece(i);
                if (piece == 1)
                    redKing = true;
                else if (piece == -1)
                    blackKing = true;
            }

            if (!redKing && blackKing)
                return GameOutcome.BlackWin;
            if (redKing && !blackKing)
                return GameOutcome.RedWin;
            return GameOutcome.Unknown;
        }

        private void AddGameResult(GameOutcome result)
        {
            LoadedGames++;
            switch (result)
            {
                case GameOutcome.RedWin:
                    RedWinGames++;
                    break;
                case GameOutcome.BlackWin:
                    BlackWinGames++;
                    break;
                case GameOutcome.Draw:
                    DrawGames++;
                    break;
                default:
                    UnknownGames++;
                    break;
            }
        }

        private void AddMove(ulong hash, Move move, bool redToMove, GameOutcome result)
        {
            if (!_entries.TryGetValue(hash, out var moves))
            {
                moves = new Dictionary<Move, MasterMoveKnowledge>();
                _entries[hash] = moves;
            }

            moves.TryGetValue(move, out var existing);
            moves[move] = existing.Add(move, redToMove, result);
        }

        private static List<string> ParseUciMoves(string text)
        {
            var moves = new List<string>();
            foreach (string rawToken in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                string token = rawToken.Trim();
                if (token.Length == 0 || token[0] == '[' || token.Contains('.'))
                    continue;

                token = token.Trim('{', '}', '(', ')');
                if (token is "1-0" or "0-1" or "1/2-1/2" or "*" or "red" or "black" or "draw")
                    continue;

                if (IsUciMove(token))
                    moves.Add(token[..4]);
            }

            return moves;
        }

        private static bool TryParseTag(string line, out string name, out string value)
        {
            name = string.Empty;
            value = string.Empty;
            if (!line.StartsWith('[') || !line.EndsWith(']'))
                return false;

            int space = line.IndexOf(' ');
            if (space <= 1)
                return false;

            name = line[1..space];
            int firstQuote = line.IndexOf('"', space);
            int lastQuote = line.LastIndexOf('"');
            if (firstQuote < 0 || lastQuote <= firstQuote)
                return false;

            value = line[(firstQuote + 1)..lastQuote];
            return true;
        }

        private static GameOutcome ParseResult(string? result)
        {
            return result switch
            {
                "1-0" => GameOutcome.RedWin,
                "0-1" => GameOutcome.BlackWin,
                "1/2-1/2" => GameOutcome.Draw,
                _ => GameOutcome.Unknown
            };
        }

        private static bool IsUciMove(string token)
        {
            if (token.Length < 4)
                return false;

            return token[0] is >= 'a' and <= 'i'
                && token[1] is >= '0' and <= '9'
                && token[2] is >= 'a' and <= 'i'
                && token[3] is >= '0' and <= '9';
        }

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

        public enum GameOutcome
        {
            Unknown,
            RedWin,
            BlackWin,
            Draw
        }

        public readonly record struct MasterMoveKnowledge(
            Move Move,
            int Count,
            int RedWins,
            int BlackWins,
            int Draws,
            int Unknown,
            bool SideToMove)
        {
            public int ScoreFromSideToMove
            {
                get
                {
                    int wins = SideToMove ? RedWins : BlackWins;
                    int losses = SideToMove ? BlackWins : RedWins;
                    return (wins - losses) * 2_000 / Math.Max(1, Count);
                }
            }

            public MasterMoveKnowledge Add(Move move, bool redToMove, GameOutcome result)
            {
                int redWins = RedWins;
                int blackWins = BlackWins;
                int draws = Draws;
                int unknown = Unknown;

                switch (result)
                {
                    case GameOutcome.RedWin:
                        redWins++;
                        break;
                    case GameOutcome.BlackWin:
                        blackWins++;
                        break;
                    case GameOutcome.Draw:
                        draws++;
                        break;
                    default:
                        unknown++;
                        break;
                }

                return new MasterMoveKnowledge(move, Count + 1, redWins, blackWins, draws, unknown, redToMove);
            }
        }

        private sealed class MasterKnowledgeCache
        {
            public int MaxPly { get; set; }
            public List<MasterKnowledgePosition> Positions { get; set; } = new();
        }

        private sealed class MasterKnowledgePosition
        {
            public string Hash { get; set; } = string.Empty;
            public List<MasterKnowledgeMove> Moves { get; set; } = new();
        }

        private sealed class MasterKnowledgeMove
        {
            public string Move { get; set; } = string.Empty;
            public int Count { get; set; }
            public int RedWins { get; set; }
            public int BlackWins { get; set; }
            public int Draws { get; set; }
            public int Unknown { get; set; }
            public bool SideToMove { get; set; }
        }
    }
}
