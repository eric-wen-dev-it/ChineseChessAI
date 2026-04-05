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

        public void RandomizePersonality()
        {
            var rnd = Random.Shared;
            Temperature = 0.1 + rnd.NextDouble() * 1.9;    // 随机温度
            Cpuct = 1.0 + rnd.NextDouble() * 4.0;          // 随机好奇心
            MctsSimulations = 100 + rnd.Next(701);         // 随机思考深度
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
                    }
                    catch
                    {
                        _agents = new List<AgentMetadata>();
                    }
                }

                // 补齐到 populationSize
                if (_agents.Count < populationSize)
                {
                    for (int i = _agents.Count; i < populationSize; i++)
                    {
                        var agent = new AgentMetadata
                        {
                            Id = i,
                            ModelPath = Path.Combine(_modelsDir, $"agent_{i}.pt")
                        };
                        agent.RandomizePersonality(); // 【新增】为新智能体注入个性基因
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
                // 策略：随机挑选一个，然后在它 ELO 附近的范围内挑选另一个（或者纯随机）
                // 既然用户说是“随机匹配”，那就先纯随机
                var rnd = Random.Shared;
                int idx1 = rnd.Next(_agents.Count);
                int idx2 = rnd.Next(_agents.Count);
                while (idx1 == idx2) idx2 = rnd.Next(_agents.Count);

                return (_agents[idx1], _agents[idx2]);
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
