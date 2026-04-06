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

        public double Temperature { get; set; } = 1.0;
        public double Cpuct { get; set; } = 2.5;
        public int MctsSimulations { get; set; } = 400;

        public void RandomizePersonality(Random? customRnd = null)
        {
            var rnd = customRnd ?? Random.Shared;
            Temperature = 0.1 + rnd.NextDouble() * 1.9;
            Cpuct = 1.0 + rnd.NextDouble() * 4.0;
            MctsSimulations = 100 + rnd.Next(701);
        }
    }

    public class LeagueManager
    {
        private readonly string _metadataPath;
        private readonly string _modelsDir;
        private List<AgentMetadata> _agents = new List<AgentMetadata>();
        
        // 【公平性核心】：全局参赛待办队列
        private List<int> _waitList = new List<int>();
        private readonly object _lock = new object();

        public LeagueManager(int populationSize = 10000)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            _metadataPath = Path.Combine(baseDir, "data", "league_metadata.json");
            _modelsDir = Path.Combine(baseDir, "data", "models", "league");
            if (!Directory.Exists(_modelsDir)) Directory.CreateDirectory(_modelsDir);
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
                    catch { _agents = new List<AgentMetadata>(); }
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

                // 初始化待赛队列：包含所有智能体 ID
                _waitList = _agents.Select(a => a.Id).OrderBy(_ => Random.Shared.Next()).ToList();
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

        public AgentMetadata? GetAgentMeta(int id)
        {
            lock (_lock) return _agents.FirstOrDefault(a => a.Id == id);
        }

        public (AgentMetadata, AgentMetadata) PickMatch()
        {
            lock (_lock)
            {
                // 1. 如果待赛队列空了，重置它，开始新的一轮大循环
                if (_waitList.Count == 0)
                {
                    _waitList = _agents.Select(a => a.Id).OrderBy(_ => Random.Shared.Next()).ToList();
                }

                // 2. 取出队列头部的智能体作为 Agent A (确保他这一轮一定参赛)
                int idA = _waitList[0];
                _waitList.RemoveAt(0);
                var agentA = _agents.First(a => a.Id == idA);

                // 3. 在全量池中寻找一个 ELO 最接近的对手 Agent B
                // 优先从还没打过这一轮的人（待赛队列）里找，如果待赛队列里没人了，则从全量池找
                var searchPool = _waitList.Count > 0 ? _waitList : _agents.Select(a => a.Id).ToList();
                
                var agentBId = searchPool
                    .Where(id => id != idA)
                    .OrderBy(id => Math.Abs(_agents.First(a => a.Id == id).Elo - agentA.Elo))
                    .Take(10) // 在最接近的 10 人中随机选一个，增加多样性
                    .OrderBy(_ => Random.Shared.Next())
                    .First();

                // 如果 Agent B 也在待赛队列中，同步将其移除，防止他这一轮打两次
                _waitList.Remove(agentBId);

                var agentB = _agents.First(a => a.Id == agentBId);
                return (agentA, agentB);
            }
        }

        public List<AgentMetadata> GetTopAgents(int count = 10)
        {
            lock (_lock) return _agents.OrderByDescending(a => a.Elo).Take(count).ToList();
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
                double expectedScore = 1.0 / (1.0 + Math.Pow(10, (opponentElo - agent.Elo) / 400.0));
                double actualScore = (result + 1.0) / 2.0;
                agent.Elo += 32 * (actualScore - expectedScore);
                agent.LastActive = DateTime.Now;
            }
        }
    }
}
