using System.IO;
using System.Text.Json;

namespace ChineseChessAI.Training
{
    public class AgentMetadata
    {
        public int Id
        {
            get; set;
        }
        public double Elo { get; set; } = 1500;
        public int GamesPlayed { get; set; } = 0;
        public int Wins { get; set; } = 0;
        public int Losses { get; set; } = 0;
        public int Draws { get; set; } = 0;
        public string ModelPath { get; set; } = "";
        public DateTime LastActive { get; set; } = DateTime.Now;
        public int Generation { get; set; } = 0;
        public int ParentId { get; set; } = -1;

        public double Temperature { get; set; } = 1.0;
        public double Cpuct { get; set; } = 1.6;
        public int MctsSimulations { get; set; } = 400;
        public string EngineKind { get; set; } = "Mcts";
        public int TraditionalDepth { get; set; } = 5;

        public void RandomizePersonality(Random? customRnd = null)
        {
            var rnd = customRnd ?? Random.Shared;
            Temperature = 0.1 + rnd.NextDouble() * 1.9;
            Cpuct = 1.0 + rnd.NextDouble() * 2.0;
            MctsSimulations = 100 + rnd.Next(701);
            EngineKind = "Mcts";
            TraditionalDepth = 4 + rnd.Next(3);
        }

        public void MutateFromParent(AgentMetadata parent, Random? customRnd = null, bool wideMutation = false)
        {
            var rnd = customRnd ?? Random.Shared;
            double tempSpan = wideMutation ? 0.55 : 0.25;
            double cpuctSpan = wideMutation ? 0.9 : 0.45;
            int simSpan = wideMutation ? 180 : 90;

            Temperature = Math.Clamp(parent.Temperature + ((rnd.NextDouble() * 2.0) - 1.0) * tempSpan, 0.1, 2.0);
            Cpuct = Math.Clamp(parent.Cpuct + ((rnd.NextDouble() * 2.0) - 1.0) * cpuctSpan, 0.8, 3.0);
            MctsSimulations = Math.Clamp(parent.MctsSimulations + rnd.Next(-simSpan, simSpan + 1), 100, 800);
            EngineKind = parent.EngineKind;
            TraditionalDepth = Math.Clamp(parent.TraditionalDepth + rnd.Next(-1, 2), 3, 8);
        }

        public void ResetCompetitiveState(double startingElo = 1500, int generation = 0, int parentId = -1)
        {
            Elo = startingElo;
            GamesPlayed = 0;
            Wins = 0;
            Losses = 0;
            Draws = 0;
            LastActive = DateTime.Now;
            Generation = generation;
            ParentId = parentId;
        }
    }

    public sealed class PopulationRefreshResult
    {
        public int EliteKept
        {
            get; init;
        }
        public int ContenderKept
        {
            get; init;
        }
        public int DiverseKept
        {
            get; init;
        }
        public int Replaced
        {
            get; init;
        }
        public int OffspringCreated
        {
            get; init;
        }
        public int ImmigrantsCreated
        {
            get; init;
        }
        public List<int> ReplacedAgentIds { get; init; } = new List<int>();
        public List<string> PreviewLines { get; init; } = new List<string>();
    }

    public class LeagueManager
    {
        private readonly string _metadataPath;
        private readonly string _modelsDir;
        private List<AgentMetadata> _agents = new List<AgentMetadata>();
        private List<int> _waitList = new List<int>();
        private readonly object _lock = new object();

        private readonly int _traditionalAgentCount;

        public LeagueManager(int populationSize = 10000, int traditionalAgentCount = 0)
        {
            _traditionalAgentCount = Math.Clamp(traditionalAgentCount, 0, Math.Max(0, populationSize - 1));
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            _metadataPath = Path.Combine(baseDir, "data", "league_metadata.json");
            _modelsDir = Path.Combine(baseDir, "data", "models", "league");
            if (!Directory.Exists(_modelsDir))
                Directory.CreateDirectory(_modelsDir);
            LoadMetadata(populationSize);
        }

        private void LoadMetadata(int populationSize)
        {
            lock (_lock)
            {
                if (File.Exists(_metadataPath))
                {
                    try
                    {
                        string json = File.ReadAllText(_metadataPath);
                        _agents = JsonSerializer.Deserialize<List<AgentMetadata>>(json) ?? new List<AgentMetadata>();
                    }
                    catch
                    {
                        try
                        {
                            string corruptPath = _metadataPath + $".corrupt_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
                            File.Move(_metadataPath, corruptPath, overwrite: false);
                        }
                        catch
                        {
                        }

                        _agents = new List<AgentMetadata>();
                    }
                }

                if (_agents.Count < populationSize)
                {
                    var seedRnd = new Random();
                    for (int i = _agents.Count; i < populationSize; i++)
                    {
                        var agent = new AgentMetadata { Id = i, ModelPath = Path.Combine(_modelsDir, $"agent_{i}.pt") };
                        agent.RandomizePersonality(seedRnd);
                        _agents.Add(agent);
                    }

                    SaveMetadata();
                }
                else if (_agents.Count > populationSize)
                {
                    _agents = _agents.OrderBy(a => a.Id).Take(populationSize).ToList();
                }

                ConfigureTraditionalAgents(populationSize);
                _waitList = _agents.Select(a => a.Id).OrderBy(_ => Random.Shared.Next()).ToList();
            }
        }

        private void ConfigureTraditionalAgents(int populationSize)
        {
            int firstTraditionalId = Math.Max(0, populationSize - _traditionalAgentCount);
            var rnd = new Random();
            foreach (var agent in _agents)
            {
                if (agent.Id >= firstTraditionalId)
                {
                    agent.EngineKind = "Traditional";
                    agent.TraditionalDepth = Math.Clamp(agent.TraditionalDepth, 4, 8);
                    agent.MctsSimulations = 0;
                    agent.Cpuct = 1.6;
                    agent.Temperature = 0.05;
                    if (string.IsNullOrWhiteSpace(agent.ModelPath))
                        agent.ModelPath = Path.Combine(_modelsDir, $"agent_{agent.Id}.pt");
                }
                else if (string.Equals(agent.EngineKind, "Traditional", StringComparison.OrdinalIgnoreCase))
                {
                    agent.RandomizePersonality(rnd);
                    agent.ModelPath = Path.Combine(_modelsDir, $"agent_{agent.Id}.pt");
                }
            }

            SaveMetadata();
        }

        public void SaveMetadata()
        {
            lock (_lock)
            {
                string json = JsonSerializer.Serialize(_agents, new JsonSerializerOptions { WriteIndented = true });
                string? directory = Path.GetDirectoryName(_metadataPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                    Directory.CreateDirectory(directory);

                string tempPath = _metadataPath + $".tmp_{Guid.NewGuid():N}";
                File.WriteAllText(tempPath, json);
                File.Move(tempPath, _metadataPath, overwrite: true);
            }
        }

        public AgentMetadata? GetAgentMeta(int id)
        {
            lock (_lock)
                return _agents.FirstOrDefault(a => a.Id == id);
        }

        public int GetPopulationSize()
        {
            lock (_lock)
                return _agents.Count;
        }

        public List<int> GetAllAgentIds()
        {
            lock (_lock)
                return _agents.Select(a => a.Id).OrderBy(id => id).ToList();
        }

        public (AgentMetadata, AgentMetadata) PickMatch()
        {
            if (TryPickMatch(Array.Empty<int>(), out var agentA, out var agentB))
            {
                return (agentA, agentB);
            }

            throw new InvalidOperationException("No eligible match is available.");
        }

        public bool TryPickMatch(IReadOnlyCollection<int> unavailableAgentIds, out AgentMetadata agentA, out AgentMetadata agentB)
        {
            lock (_lock)
            {
                agentA = null!;
                agentB = null!;

                if (_waitList.Count == 0)
                {
                    _waitList = _agents.Select(a => a.Id).OrderBy(_ => Random.Shared.Next()).ToList();
                }

                HashSet<int>? unavailable = unavailableAgentIds.Count > 0
                    ? new HashSet<int>(unavailableAgentIds)
                    : null;

                var availableAgents = _agents
                    .Where(a => unavailable == null || !unavailable.Contains(a.Id))
                    .ToList();

                if (availableAgents.Count < 2)
                {
                    return false;
                }

                int waitListIndex = _waitList.FindIndex(id => unavailable == null || !unavailable.Contains(id));
                int idA;
                if (waitListIndex >= 0)
                {
                    idA = _waitList[waitListIndex];
                    _waitList.RemoveAt(waitListIndex);
                }
                else
                {
                    idA = availableAgents
                        .OrderBy(_ => Random.Shared.Next())
                        .First()
                        .Id;
                }

                agentA = _agents.First(a => a.Id == idA);
                double agentAElo = agentA.Elo;

                var searchPool = _waitList
                    .Where(id => id != idA && (unavailable == null || !unavailable.Contains(id)))
                    .Distinct()
                    .ToList();

                if (searchPool.Count == 0)
                {
                    searchPool = availableAgents
                        .Select(a => a.Id)
                        .Where(id => id != idA)
                        .ToList();
                }

                if (searchPool.Count == 0)
                {
                    return false;
                }

                int agentBId = searchPool
                    .OrderBy(id => Math.Abs(_agents.First(a => a.Id == id).Elo - agentAElo))
                    .Take(10)
                    .OrderBy(_ => Random.Shared.Next())
                    .First();

                _waitList.Remove(agentBId);
                agentB = _agents.First(a => a.Id == agentBId);
                return true;
            }
        }

        public List<AgentMetadata> GetTopAgents(int count = 10)
        {
            lock (_lock)
                return _agents.OrderByDescending(a => a.Elo).Take(count).ToList();
        }

        public List<AgentMetadata> GetTopNeuralAgents(int count = 10)
        {
            lock (_lock)
            {
                return _agents
                    .Where(a => !string.Equals(a.EngineKind, "Traditional", StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(a => a.Elo)
                    .Take(count)
                    .ToList();
            }
        }

        public PopulationRefreshResult RefreshPopulation(
            int eliteCount,
            int contenderKeepCount,
            int diverseKeepCount,
            int parentPoolSize,
            int immigrantCount)
        {
            lock (_lock)
            {
                int populationSize = _agents.Count;
                if (populationSize < 6)
                {
                    return new PopulationRefreshResult();
                }

                eliteCount = Math.Clamp(eliteCount, 1, Math.Max(1, populationSize - 1));
                contenderKeepCount = Math.Clamp(contenderKeepCount, 0, Math.Max(0, populationSize - eliteCount - 1));
                diverseKeepCount = Math.Clamp(diverseKeepCount, 0, Math.Max(0, populationSize - eliteCount - contenderKeepCount - 1));

                int survivorTarget = eliteCount + contenderKeepCount + diverseKeepCount;
                if (survivorTarget >= populationSize)
                {
                    diverseKeepCount = Math.Max(0, populationSize - eliteCount - contenderKeepCount - 1);
                    survivorTarget = eliteCount + contenderKeepCount + diverseKeepCount;
                }

                var rnd = Random.Shared;
                var traditionalAgents = _agents
                    .Where(a => string.Equals(a.EngineKind, "Traditional", StringComparison.OrdinalIgnoreCase))
                    .ToList();

                var ranked = _agents
                    .Where(a => !string.Equals(a.EngineKind, "Traditional", StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(a => a.Elo)
                    .ThenByDescending(a => a.Wins)
                    .ThenBy(a => a.Id)
                    .ToList();
                populationSize = ranked.Count;
                if (populationSize < 6)
                {
                    return new PopulationRefreshResult();
                }

                eliteCount = Math.Clamp(eliteCount, 1, Math.Max(1, populationSize - 1));
                contenderKeepCount = Math.Clamp(contenderKeepCount, 0, Math.Max(0, populationSize - eliteCount - 1));
                diverseKeepCount = Math.Clamp(diverseKeepCount, 0, Math.Max(0, populationSize - eliteCount - contenderKeepCount - 1));

                var elites = ranked.Take(eliteCount).ToList();
                var contenders = ranked.Skip(eliteCount).Take(contenderKeepCount).ToList();
                var diversePool = ranked.Skip(eliteCount + contenderKeepCount).ToList();
                var diverseKeepers = diversePool
                    .OrderBy(_ => rnd.Next())
                    .Take(diverseKeepCount)
                    .ToList();

                var survivorIds = new HashSet<int>(elites.Select(a => a.Id));
                foreach (var contender in contenders)
                    survivorIds.Add(contender.Id);
                foreach (var keeper in diverseKeepers)
                    survivorIds.Add(keeper.Id);

                var replacements = ranked.Where(a => !survivorIds.Contains(a.Id)).ToList();
                if (replacements.Count == 0)
                {
                    return new PopulationRefreshResult
                    {
                        EliteKept = elites.Count,
                        ContenderKept = contenders.Count,
                        DiverseKept = diverseKeepers.Count
                    };
                }

                var survivors = elites
                    .Concat(contenders)
                    .Concat(diverseKeepers)
                    .OrderByDescending(a => a.Elo)
                    .ThenByDescending(a => a.Wins)
                    .ThenBy(a => a.Id)
                    .ToList();

                parentPoolSize = Math.Clamp(parentPoolSize, 1, Math.Min(10, survivors.Count));
                var parentPool = survivors.Take(parentPoolSize).ToList();

                int actualImmigrantCount = Math.Clamp(immigrantCount, 0, replacements.Count);
                int actualOffspringCount = replacements.Count - actualImmigrantCount;

                var previewLines = new List<string>();
                for (int i = 0; i < replacements.Count; i++)
                {
                    var replacement = replacements[i];
                    var parent = PickWeightedParent(parentPool, rnd);
                    bool isImmigrant = i >= actualOffspringCount;

                    CopyModelFromParent(parent.ModelPath, replacement.ModelPath);

                    if (isImmigrant)
                    {
                        replacement.RandomizePersonality(rnd);
                    }
                    else
                    {
                        replacement.MutateFromParent(parent, rnd);
                    }

                    replacement.ResetCompetitiveState(
                        startingElo: 1500,
                        generation: parent.Generation + 1,
                        parentId: parent.Id);

                    previewLines.Add(
                        $"{replacement.Id}<-{parent.Id} {(isImmigrant ? "immigrant" : "offspring")} " +
                        $"DNA:S{replacement.MctsSimulations}/C{replacement.Cpuct:F1}/T{replacement.Temperature:F1}");
                }

                foreach (var traditionalAgent in traditionalAgents)
                {
                    if (!_agents.Any(a => a.Id == traditionalAgent.Id))
                        _agents.Add(traditionalAgent);
                }

                _agents = _agents.OrderBy(a => a.Id).ToList();
                _waitList = _agents.Select(a => a.Id).OrderBy(_ => rnd.Next()).ToList();
                SaveMetadata();

                return new PopulationRefreshResult
                {
                    EliteKept = elites.Count,
                    ContenderKept = contenders.Count,
                    DiverseKept = diverseKeepers.Count,
                    Replaced = replacements.Count,
                    OffspringCreated = actualOffspringCount,
                    ImmigrantsCreated = actualImmigrantCount,
                    ReplacedAgentIds = replacements.Select(a => a.Id).ToList(),
                    PreviewLines = previewLines.Take(8).ToList()
                };
            }
        }

        public void UpdateResult(int agentId, float result, double opponentElo)
        {
            lock (_lock)
            {
                var agent = _agents.FirstOrDefault(a => a.Id == agentId);
                if (agent == null)
                    return;
                agent.GamesPlayed++;
                if (result > 0.001f)
                    agent.Wins++;
                else if (result < -0.001f)
                    agent.Losses++;
                else
                    agent.Draws++;
                if (string.Equals(agent.EngineKind, "Traditional", StringComparison.OrdinalIgnoreCase))
                {
                    agent.LastActive = DateTime.Now;
                    return;
                }

                double expectedScore = 1.0 / (1.0 + Math.Pow(10, (opponentElo - agent.Elo) / 400.0));
                double actualScore = (result + 1.0) / 2.0;
                double kFactor = agent.GamesPlayed <= 20 ? 48.0 : 32.0;
                agent.Elo += kFactor * (actualScore - expectedScore);
                agent.LastActive = DateTime.Now;
            }
        }

        private static AgentMetadata PickWeightedParent(List<AgentMetadata> parentPool, Random rnd)
        {
            int totalWeight = 0;
            for (int i = 0; i < parentPool.Count; i++)
            {
                totalWeight += parentPool.Count - i;
            }

            int roll = rnd.Next(totalWeight);
            int cumulative = 0;
            for (int i = 0; i < parentPool.Count; i++)
            {
                cumulative += parentPool.Count - i;
                if (roll < cumulative)
                {
                    return parentPool[i];
                }
            }

            return parentPool[0];
        }

        private static void CopyModelFromParent(string parentPath, string childPath)
        {
            // Offspring inherit weights only; stale optimizer/LR schedule files are removed.
            string[] companionSuffixes = { ".optim", ".optim.json" };

            if (File.Exists(parentPath))
            {
                string tempChildPath = childPath + $".tmp_{Guid.NewGuid():N}";
                File.Copy(parentPath, tempChildPath, overwrite: true);
                File.Move(tempChildPath, childPath, overwrite: true);
                foreach (string suffix in companionSuffixes)
                {
                    string childCompanion = childPath + suffix;
                    if (File.Exists(childCompanion))
                        File.Delete(childCompanion);
                }
                return;
            }
        }
    }
}
