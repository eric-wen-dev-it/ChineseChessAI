using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace ChineseChessAI.Training
{
    public class AgentMetadata
    {
        public int Id { get; set; }
        public double Elo { get; set; } = 1500;
        public int GamesPlayed { get; set; } = 0;
        public int Wins { get; set; } = 0;
        public int Losses { get; set; } = 0;
        public int Draws { get; set; } = 0;
        public string ModelPath { get; set; } = "";
        public DateTime LastActive { get; set; } = DateTime.Now;

        // --- 个性化基因 (性格超参数) ---
        public double Temperature { get; set; } = 1.0;     // 探索温度 [0.1, 2.0]
        public double Cpuct { get; set; } = 2.5;           // 探索常数 [1.0, 5.0]
        public int MctsSimulations { get; set; } = 400;    // 模拟次数 [100, 800]

        public void RandomizePersonality(Random? customRnd = null)
        {
            var rnd = customRnd ?? Random.Shared;
            Temperature = 0.1 + rnd.NextDouble() * 1.9;    // 随机温度 [0.1, 2.0]
            Cpuct = 1.0 + rnd.NextDouble() * 4.0;          // 随机好奇心 [1.0, 5.0]
            MctsSimulations = 100 + rnd.Next(701);         // 随机思考深度 [100, 800]
        }
    }

    public class LeagueManager
    {
        private readonly string _metadataPath;
        private readonly string _modelsDir;
        private List<AgentMetadata> _agents = new List<AgentMetadata>();
        private readonly object _lock = new object();

        public LeagueManager(int populationSize = 10000)
        {
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
                        
                        // 【核心增强】：不仅检查默认值，还要检查是否发生了“随机数同步（即不同智能体性格一模一样）”
                        bool needFix = _agents.Count > 1 && 
                                       _agents.Take(10).All(a => a.MctsSimulations == _agents[0].MctsSimulations && a.Cpuct == _agents[0].Cpuct);

                        if (needFix || (_agents.Count > 0 && _agents.All(a => a.MctsSimulations == 400 && a.Cpuct == 2.5)))
                        {
                            var fixRnd = new Random();
                            foreach (var agent in _agents) agent.RandomizePersonality(fixRnd);
                            SaveMetadata();
                        }
                    }
                    catch
                    {
                        _agents = new List<AgentMetadata>();
                    }
                }

                // 补齐到 populationSize
                if (_agents.Count < populationSize)
                {
                    // 【核心修复】：使用同一个随机数实例连续生成，防止高频创建导致的种子重合
                    var seedRnd = new Random(); 
                    for (int i = _agents.Count; i < populationSize; i++)
                    {
                        var agent = new AgentMetadata
                        {
                            Id = i,
                            ModelPath = Path.Combine(_modelsDir, $"agent_{i}.pt")
                        };
                        agent.RandomizePersonality(seedRnd); // 注入唯一的随机实例
                        _agents.Add(agent);
                    }
                    SaveMetadata();
                }
            }
        }

        public void SaveMetadata()
        {
            lock (_lock)
            {
                string json = JsonSerializer.Serialize(_agents, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_metadataPath, json);
            }
        }

        public (AgentMetadata, AgentMetadata) PickMatch()
        {
            lock (_lock)
            {
                var rnd = Random.Shared;
                int idx1 = rnd.Next(_agents.Count);
                var agentA = _agents[idx1];

                // 【核心改进】：ELO 邻近匹配策略 (±300 范围内挑选)
                var candidates = _agents.Where(a => a.Id != agentA.Id && Math.Abs(a.Elo - agentA.Elo) < 300).ToList();
                
                AgentMetadata agentB;
                if (candidates.Count > 0)
                {
                    agentB = candidates[rnd.Next(candidates.Count)];
                }
                else
                {
                    // 如果范围内没人，则选最接近的一个
                    agentB = _agents.Where(a => a.Id != agentA.Id)
                                    .OrderBy(a => Math.Abs(a.Elo - agentA.Elo))
                                    .First();
                }

                return (agentA, agentB);
            }
        }

        public List<AgentMetadata> GetTopAgents(int count = 10)
        {
            lock (_lock)
            {
                return _agents.OrderByDescending(a => a.Elo).Take(count).ToList();
            }
        }

        public void UpdateResult(int agentId, float result, double opponentElo)
        {
            lock (_lock)
            {
                var agent = _agents.FirstOrDefault(a => a.Id == agentId);
                if (agent == null) return;

                agent.GamesPlayed++;
                if (result > 0.5f) agent.Wins++;
                else if (result < -0.5f) agent.Losses++;
                else agent.Draws++;

                // K-factor = 32
                double expectedScore = 1.0 / (1.0 + Math.Pow(10, (opponentElo - agent.Elo) / 400.0));
                double actualScore = (result + 1.0) / 2.0; // [-1, 1] -> [0, 1]
                agent.Elo += 32 * (actualScore - expectedScore);
                agent.LastActive = DateTime.Now;
            }
        }
    }
}
