import { useState } from "react";

// Mock data for prototype
const MOCK_DATA = {
  lastUpdate: "Nov 16, 09:00",
  fresh: {
    score: 57,
    band: "amber" as const,
    bullets: [
      "2 high-risk days (Tue, Fri)",
      "Driven by beam-sea Poole→Jersey sailings",
    ],
  },
  fuel: {
    score: 34,
    band: "green" as const,
    bullets: ["Mostly green week", "One amber day (Tue, Poole→Jersey)"],
  },
  recommendedPlan: {
    policy: "Bring forward 10% of Fresh, 3% of Fuel",
    fresh: { baseline: 57, scenario: 37, delta: -20, hoursAvoided: 14 },
    fuel: { baseline: 34, scenario: 24, delta: -10, trailersAvoided: 2 },
  },
  days: [
    { name: "SAT", date: "15 NOV", fresh: 2, fuel: 0, band: "green" as const },
    { name: "SUN", date: "16 NOV", fresh: 3, fuel: 0, band: "green" as const },
    { name: "MON", date: "17 NOV", fresh: 1, fuel: 0, band: "green" as const },
    { name: "TUE", date: "18 NOV", fresh: 4, fuel: 0, band: "green" as const },
    { name: "WED", date: "19 NOV", fresh: 3, fuel: 0, band: "green" as const },
    { name: "THU", date: "20 NOV", fresh: 3, fuel: 0, band: "green" as const },
    { name: "FRI", date: "21 NOV", fresh: 19, fuel: 0, band: "green" as const },
  ],
  sailings: [
    {
      time: "06:00",
      id: "Condor123",
      route: "Poole→Jersey",
      risk: 71,
      freshExp: 30,
      fuelExp: 0,
    },
    {
      time: "14:00",
      id: "Condor456",
      route: "Portsmouth→Jsy",
      risk: 65,
      freshExp: 20,
      fuelExp: 12,
    },
    {
      time: "18:00",
      id: "Condor789",
      route: "Guernsey→Jsy",
      risk: 45,
      freshExp: 25,
      fuelExp: 8,
    },
  ],
  scenarios: {
    A: {
      name: "Conservative",
      description: "Bring forward 10% of Fresh, 3% of Fuel",
      fresh: { baseline: 57, scenario: 37, delta: -20, hoursAvoided: 14 },
      fuel: { baseline: 34, scenario: 24, delta: -10, trailersAvoided: 2 },
      sailings: [
        { date: "18 NOV", time: "06:00", id: "Condor123", route: "Poole→Jersey", freshDelta: -8, fuelDelta: 0, action: "Bring forward 10%" },
        { date: "18 NOV", time: "14:00", id: "Condor456", route: "Portsmouth→Jsy", freshDelta: -6, fuelDelta: -5, action: "Bring forward 10% Fresh, 3% Fuel" },
        { date: "21 NOV", time: "06:00", id: "Condor789", route: "Guernsey→Jsy", freshDelta: -6, fuelDelta: -5, action: "Bring forward 10% Fresh, 3% Fuel" },
      ],
    },
    B: {
      name: "Aggressive",
      description: "Bring forward 20% of Fresh, 10% of Fuel",
      fresh: { baseline: 57, scenario: 27, delta: -30, hoursAvoided: 21 },
      fuel: { baseline: 34, scenario: 14, delta: -20, trailersAvoided: 4 },
      sailings: [
        { date: "18 NOV", time: "06:00", id: "Condor123", route: "Poole→Jersey", freshDelta: -12, fuelDelta: 0, action: "Bring forward 20%" },
        { date: "18 NOV", time: "14:00", id: "Condor456", route: "Portsmouth→Jsy", freshDelta: -10, fuelDelta: -10, action: "Bring forward 20% Fresh, 10% Fuel" },
        { date: "21 NOV", time: "06:00", id: "Condor789", route: "Guernsey→Jsy", freshDelta: -8, fuelDelta: -10, action: "Bring forward 20% Fresh, 10% Fuel" },
      ],
    },
  },
};

const BAND_COLORS = {
  green: { bg: "bg-white", text: "text-green-600", border: "border-gray-200", label: "Low Risk" },
  amber: { bg: "bg-white", text: "text-amber-600", border: "border-gray-200", label: "Moderate Risk" },
  red: { bg: "bg-white", text: "text-red-600", border: "border-gray-200", label: "High Risk" },
};

function App() {
  const [activeTab, setActiveTab] = useState<"weekly" | "scenarios">("weekly");
  const [selectedDay, setSelectedDay] = useState(0);
  const [selectedScenario, setSelectedScenario] = useState<"A" | "B">("A");
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  const getSortedSailings = () => {
    const sailings = [...MOCK_DATA.scenarios[selectedScenario].sailings];
    if (!sortColumn) return sailings;

    return sailings.sort((a, b) => {
      let aVal: string | number = "";
      let bVal: string | number = "";

      switch (sortColumn) {
        case "date":
          aVal = a.date;
          bVal = b.date;
          break;
        case "time":
          aVal = a.time;
          bVal = b.time;
          break;
        case "sailing":
          aVal = a.id;
          bVal = b.id;
          break;
        case "route":
          aVal = a.route;
          bVal = b.route;
          break;
        case "freshDelta":
          aVal = a.freshDelta;
          bVal = b.freshDelta;
          break;
        case "fuelDelta":
          aVal = a.fuelDelta;
          bVal = b.fuelDelta;
          break;
        case "action":
          aVal = a.action;
          bVal = b.action;
          break;
      }

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
      }
      return sortDirection === "asc"
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-100 flex flex-col">
        <div className="p-6 border-b border-gray-100">
          <h1 className="text-2xl font-semibold text-gray-900">IslandSense</h1>
        </div>

        <div className="p-6 text-base text-gray-500">
          Last Update
          <br />
          <span className="text-gray-700 font-medium">{MOCK_DATA.lastUpdate}</span>
        </div>

        <nav className="flex-1 px-4">
          <button
            onClick={() => setActiveTab("weekly")}
            className={`w-full text-left px-4 py-3 text-lg mb-2 ${
              activeTab === "weekly"
                ? "bg-gray-100 text-gray-900 font-medium"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            Weekly Overview
          </button>
          <button
            onClick={() => setActiveTab("scenarios")}
            className={`w-full text-left px-4 py-3 text-lg ${
              activeTab === "scenarios"
                ? "bg-gray-100 text-gray-900 font-medium"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            Scenarios
          </button>
        </nav>

        <div className="p-6 border-t border-gray-100 text-sm text-gray-400">
          IslandSense v0.1
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <div className="bg-white border-b border-gray-100 px-8 py-4 flex justify-between items-center">
          <div>
            <h2 className="text-3xl font-semibold text-gray-900">
              {activeTab === "weekly" ? "Weekly Overview" : "Scenarios"}
            </h2>
            <p className="text-lg text-gray-500">
              {activeTab === "weekly"
                ? "Per-Sailing Disruption & 7-Day Early Warning"
                : "What if & Suggest Actions"}
            </p>
          </div>
          <div className="flex gap-3">
            <button className="px-4 py-2 text-base text-gray-600 hover:bg-gray-50 border border-gray-200 rounded flex items-center gap-2">
              <span>📄</span> PDF
            </button>
            <button className="px-4 py-2 text-base text-gray-600 hover:bg-gray-50 border border-gray-200 rounded flex items-center gap-2">
              <span>✉️</span> Email
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto py-8 px-64 bg-gray-50">
          {activeTab === "weekly" && (
            <div className="space-y-8">
              {/* Category Summary Cards */}
              <div className="grid grid-cols-2 gap-6">
                {/* Fresh Card */}
                <div className="p-6 bg-white border border-gray-200">
                  <div className="text-xl font-semibold text-gray-900 mb-3">
                    FRESH
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold text-gray-900">
                      {MOCK_DATA.fresh.score}
                    </span>
                    <span className="text-2xl text-gray-400">/100</span>
                  </div>
                  <div
                    className={`text-lg mt-2 ${BAND_COLORS[MOCK_DATA.fresh.band].text}`}
                  >
                    {BAND_COLORS[MOCK_DATA.fresh.band].label}
                  </div>
                  <ul className="mt-4 space-y-1 text-base text-gray-500">
                    {MOCK_DATA.fresh.bullets.map((b, i) => (
                      <li key={i}>• {b}</li>
                    ))}
                  </ul>
                </div>

                {/* Fuel Card */}
                <div className="p-6 bg-white border border-gray-200">
                  <div className="text-xl font-semibold text-gray-900 mb-3">
                    FUEL
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold text-gray-900">
                      {MOCK_DATA.fuel.score}
                    </span>
                    <span className="text-2xl text-gray-400">/100</span>
                  </div>
                  <div
                    className={`text-lg mt-2 ${BAND_COLORS[MOCK_DATA.fuel.band].text}`}
                  >
                    {BAND_COLORS[MOCK_DATA.fuel.band].label}
                  </div>
                  <ul className="mt-4 space-y-1 text-base text-gray-500">
                    {MOCK_DATA.fuel.bullets.map((b, i) => (
                      <li key={i}>• {b}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Recommended Weekly Plan */}
              <div
                onClick={() => setActiveTab("scenarios")}
                className="p-6 bg-amber-50 border-2 border-amber-200 cursor-pointer hover:border-amber-400 hover:shadow-md transition-all"
              >
                <div className="flex justify-between items-center">
                  <div className="text-xl font-semibold text-gray-900">
                    Recommended Weekly Plan
                  </div>
                  <span className="text-base font-medium text-gray-600 hover:text-gray-900">
                    View scenarios →
                  </span>
                </div>
                <p className="mt-2 text-lg font-medium text-gray-700">
                  Policy: {MOCK_DATA.recommendedPlan.policy}
                </p>
                <div className="mt-4 grid grid-cols-2 gap-6">
                  <div>
                    <div className="text-lg">
                      <span className="text-green-600 font-semibold">
                        -{Math.abs(MOCK_DATA.recommendedPlan.fresh.delta)}%
                      </span>
                      <span className="text-gray-700 ml-2">Fresh Risk</span>
                    </div>
                    <div className="text-base text-gray-500 mt-1">
                      ~{MOCK_DATA.recommendedPlan.fresh.hoursAvoided}h shelf-gap
                      avoided
                    </div>
                  </div>
                  <div>
                    <div className="text-lg">
                      <span className="text-green-600 font-semibold">
                        -{Math.abs(MOCK_DATA.recommendedPlan.fuel.delta)}%
                      </span>
                      <span className="text-gray-700 ml-2">Fuel Risk</span>
                    </div>
                    <div className="text-base text-gray-500 mt-1">
                      ~{MOCK_DATA.recommendedPlan.fuel.trailersAvoided} trailers
                      saved
                    </div>
                  </div>
                </div>
              </div>

              {/* Daily Risk Strip */}
              <div className="bg-white p-6 border border-gray-200">
                <div className="text-xl font-semibold text-gray-900 mb-4">
                  Daily Risk by Category
                </div>
                <div className="grid grid-cols-7 gap-3">
                  {MOCK_DATA.days.map((day, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedDay(i)}
                      className={`p-4 border text-center transition-all ${
                        selectedDay === i
                          ? "border-gray-900 bg-gray-100"
                          : "border-gray-200 hover:border-gray-300 bg-white"
                      }`}
                    >
                      <div className="text-base font-medium text-gray-900">
                        {day.name}
                      </div>
                      <div className="text-base text-gray-400">{day.date}</div>
                      <div className="mt-3 text-base text-gray-600">
                        Fresh: {day.fresh} | Fuel: {day.fuel}
                      </div>
                    </button>
                  ))}
                </div>

                {/* Sailing Table (inline expansion) */}
                <div className="mt-8">
                  <div className="flex justify-between items-center mb-4">
                    <div className="text-base font-medium text-gray-900">
                      Sailings | {MOCK_DATA.days[selectedDay].name}{" "}
                      {MOCK_DATA.days[selectedDay].date}
                    </div>
                    <button className="text-sm text-gray-500 hover:text-gray-700">
                      Export CSV
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Time
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Sailing
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Route
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Disruption Risk
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Fresh Exposure
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Fuel Exposure
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {MOCK_DATA.sailings.map((s, i) => (
                          <tr
                            key={i}
                            className="border-b border-gray-100"
                          >
                            <td className="py-3 text-gray-600">{s.time}</td>
                            <td className="py-3 font-medium text-gray-900">
                              {s.id}
                            </td>
                            <td className="py-3 text-gray-600">{s.route}</td>
                            <td className="py-3">
                              <span
                                className={`font-medium ${s.risk >= 70 ? "text-red-600" : s.risk >= 40 ? "text-amber-600" : "text-gray-900"}`}
                              >
                                {s.risk}%
                              </span>
                            </td>
                            <td className="py-3 text-gray-600">
                              {s.freshExp} pal
                            </td>
                            <td className="py-3 text-gray-600">
                              {s.fuelExp} tr
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "scenarios" && (
            <div className="space-y-8">
              {/* Scenario Tabs */}
              <div className="flex gap-4">
                <button
                  onClick={() => setSelectedScenario("A")}
                  className={`px-6 py-3 text-lg font-medium border-b-2 transition-all ${
                    selectedScenario === "A"
                      ? "border-gray-900 text-gray-900"
                      : "border-transparent text-gray-500 hover:text-gray-700"
                  }`}
                >
                  Scenario A: {MOCK_DATA.scenarios.A.name}
                </button>
                <button
                  onClick={() => setSelectedScenario("B")}
                  className={`px-6 py-3 text-lg font-medium border-b-2 transition-all ${
                    selectedScenario === "B"
                      ? "border-gray-900 text-gray-900"
                      : "border-transparent text-gray-500 hover:text-gray-700"
                  }`}
                >
                  Scenario B: {MOCK_DATA.scenarios.B.name}
                </button>
              </div>

              {/* Impact Summary */}
              <div className="bg-amber-50 p-6 border-2 border-amber-200">
                <div className="text-xl font-semibold text-gray-900 mb-4">
                  Impact Summary
                </div>
                <p className="text-lg font-medium text-gray-700 mb-6">
                  Policy: {MOCK_DATA.scenarios[selectedScenario].description}
                </p>
                <div className="grid grid-cols-2 gap-6">
                  <div className="p-4 bg-white border border-gray-200">
                    <div className="text-base font-medium text-gray-900 mb-2">
                      Fresh Risk
                    </div>
                    <div className="text-5xl font-bold text-gray-900">
                      {MOCK_DATA.scenarios[selectedScenario].fresh.baseline} →{" "}
                      {MOCK_DATA.scenarios[selectedScenario].fresh.scenario}
                    </div>
                    <div className="text-lg mt-2">
                      <span className="text-amber-600">Moderate Risk</span> →{" "}
                      <span className="text-green-600">Low Risk</span>
                    </div>
                    <div className="text-base text-gray-500 mt-2">
                      ~{MOCK_DATA.scenarios[selectedScenario].fresh.hoursAvoided}h
                      shelf-gap avoided
                    </div>
                  </div>
                  <div className="p-4 bg-white border border-gray-200">
                    <div className="text-base font-medium text-gray-900 mb-2">
                      Fuel Risk
                    </div>
                    <div className="text-5xl font-bold text-gray-900">
                      {MOCK_DATA.scenarios[selectedScenario].fuel.baseline} →{" "}
                      {MOCK_DATA.scenarios[selectedScenario].fuel.scenario}
                    </div>
                    <div className="text-lg mt-2">
                      <span className="text-green-600">Low Risk</span> →{" "}
                      <span className="text-green-600">Low Risk</span>
                    </div>
                    <div className="text-base text-gray-500 mt-2">
                      ~{MOCK_DATA.scenarios[selectedScenario].fuel.trailersAvoided}{" "}
                      trailers saved
                    </div>
                  </div>
                </div>
              </div>

              {/* Sailing Actions Table */}
              <div className="bg-white p-6 border border-gray-200">
                <div className="text-xl font-semibold text-gray-900 mb-4">
                  Per-Sailing Suggested Actions
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th
                          onClick={() => handleSort("date")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Date {sortColumn === "date" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                        <th
                          onClick={() => handleSort("time")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Time {sortColumn === "time" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                        <th
                          onClick={() => handleSort("sailing")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Sailing {sortColumn === "sailing" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                        <th
                          onClick={() => handleSort("route")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Route {sortColumn === "route" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                        <th
                          onClick={() => handleSort("freshDelta")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Weekly Fresh Risk Δ {sortColumn === "freshDelta" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                        <th
                          onClick={() => handleSort("fuelDelta")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Weekly Fuel Risk Δ {sortColumn === "fuelDelta" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                        <th
                          onClick={() => handleSort("action")}
                          className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide cursor-pointer hover:text-gray-700"
                        >
                          Suggested Action {sortColumn === "action" && (sortDirection === "asc" ? "↑" : "↓")}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {getSortedSailings().map(
                        (s, i) => (
                          <tr key={i} className="border-b border-gray-100">
                            <td className="py-3 text-gray-600">{s.date}</td>
                            <td className="py-3 text-gray-600">{s.time}</td>
                            <td className="py-3 font-medium text-gray-900">
                              {s.id}
                            </td>
                            <td className="py-3 text-gray-600">{s.route}</td>
                            <td className="py-3">
                              <span className="text-green-600 font-medium">
                                {s.freshDelta}
                              </span>
                            </td>
                            <td className="py-3">
                              <span className="text-green-600 font-medium">
                                {s.fuelDelta}
                              </span>
                            </td>
                            <td className="py-3 text-gray-600">{s.action}</td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
