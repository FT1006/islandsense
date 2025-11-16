import { useState, useEffect } from "react";

interface DashboardData {
  lastUpdate: string;
  fresh: {
    score: number;
    band: "green" | "amber" | "red";
    bullets: string[];
  };
  fuel: {
    score: number;
    band: "green" | "amber" | "red";
    bullets: string[];
  };
  recommendedPlan: {
    policy: string;
    forwardFrac?: number;
    fresh: { baseline: number; scenario: number; delta: number; hoursAvoided: number };
    fuel: { baseline: number; scenario: number; delta: number; trailersAvoided: number };
  };
  days: Array<{
    name: string;
    date: string;
    fresh: number;
    fuel: number;
    band: "green" | "amber" | "red";
  }>;
  sailings: Array<{
    time: string;
    id: string;
    route: string;
    risk: number;
    freshExp: number;
    fuelExp: number;
    wotdi?: number;
    bsef?: number;
    gust?: number;
    tide?: number;
  }>;
  sailingsByDay: Record<number, Array<{
    time: string;
    id: string;
    route: string;
    risk: number;
    freshExp: number;
    fuelExp: number;
    wotdi?: number;
    bsef?: number;
    gust?: number;
    tide?: number;
  }>>;
  scenarios: {
    A: {
      name: string;
      description: string;
      forwardFrac?: number;
      fresh: { baseline: number; scenario: number; delta: number; hoursAvoided: number };
      fuel: { baseline: number; scenario: number; delta: number; trailersAvoided: number };
      sailings: Array<{
        date: string;
        time: string;
        id: string;
        route: string;
        freshDelta: number;
        fuelDelta: number;
        action: string;
      }>;
    };
    B: {
      name: string;
      description: string;
      forwardFrac?: number;
      fresh: { baseline: number; scenario: number; delta: number; hoursAvoided: number };
      fuel: { baseline: number; scenario: number; delta: number; trailersAvoided: number };
      sailings: Array<{
        date: string;
        time: string;
        id: string;
        route: string;
        freshDelta: number;
        fuelDelta: number;
        action: string;
      }>;
    };
  };
}

const BAND_COLORS = {
  green: { bg: "bg-white", text: "text-green-600", border: "border-gray-200", label: "Low Risk" },
  amber: { bg: "bg-white", text: "text-amber-600", border: "border-gray-200", label: "Moderate Risk" },
  red: { bg: "bg-white", text: "text-red-600", border: "border-gray-200", label: "High Risk" },
};

// Helper to get risk label and color from score (0-100)
const getRiskInfo = (score: number): { label: string; colorClass: string } => {
  if (score <= 20) {
    return { label: "Low Risk", colorClass: "text-green-600" };
  } else if (score <= 50) {
    return { label: "Moderate Risk", colorClass: "text-amber-600" };
  } else {
    return { label: "High Risk", colorClass: "text-red-600" };
  }
};

function App() {
  const [activeTab, setActiveTab] = useState<"weekly" | "scenarios">("weekly");
  const [selectedDay, setSelectedDay] = useState(0);
  const [selectedScenario, setSelectedScenario] = useState<"A" | "B">("A");
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/dashboard")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch dashboard data");
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  const exportSailingsCSV = () => {
    if (!data) return;
    const currentSailings = data.sailingsByDay[selectedDay] || data.sailings;
    const headers = ["Time", "Sailing", "Route", "Disruption Risk", "Fresh Exposure", "Fuel Exposure"];
    const rows = currentSailings.map((s) => [s.time, s.id, s.route, `${s.risk}%`, s.freshExp, s.fuelExp]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sailings_${data.days[selectedDay]?.date || "data"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportScenarioCSV = () => {
    if (!data) return;
    const headers = ["Date", "Time", "Sailing", "Route", "Weekly Fresh Risk Delta", "Weekly Fuel Risk Delta", "Suggested Action"];
    const rows = getSortedSailings().map((s) => [s.date, s.time, s.id, s.route, s.freshDelta, s.fuelDelta, s.action]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scenario_${selectedScenario}_actions.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getSortedSailings = () => {
    if (!data) return [];
    const sailings = [...data.scenarios[selectedScenario].sailings];
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

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-xl text-gray-600">Loading dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-xl text-red-600">Error: {error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-xl text-gray-600">No data available</div>
      </div>
    );
  }

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
          <span className="text-gray-700 font-medium">{data.lastUpdate}</span>
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
            <button
              onClick={() => window.print()}
              className="px-4 py-2 text-base text-gray-600 hover:bg-gray-50 border border-gray-200 rounded flex items-center gap-2"
            >
              <span>📄</span> PDF
            </button>
            <button
              onClick={() => {
                const subject = encodeURIComponent("IslandSense Weekly Report");
                const body = encodeURIComponent(
                  `Weekly Risk Summary:\n\nFresh Risk: ${data?.fresh.score}/100 (${data?.fresh.band})\nFuel Risk: ${data?.fuel.score}/100 (${data?.fuel.band})\n\nRecommended Action: ${data?.recommendedPlan.policy}\n\nView full dashboard for details.`
                );
                window.open(`mailto:?subject=${subject}&body=${body}`);
              }}
              className="px-4 py-2 text-base text-gray-600 hover:bg-gray-50 border border-gray-200 rounded flex items-center gap-2"
            >
              <span>✉️</span> Email
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto py-8 px-4 sm:px-8 lg:px-16 xl:px-32 2xl:px-64 print:px-4 bg-gray-50">
          {activeTab === "weekly" && (
            <div className="space-y-8">
              {/* Category Summary Cards */}
              <div className="grid grid-cols-2 gap-6">
                {/* Fresh Card */}
                <div className="p-6 bg-white border border-gray-200">
                  <div className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-3">
                    FRESH WEEKLY RISK SCORE
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold text-gray-900">
                      {data.fresh.score}
                    </span>
                    <span className="text-2xl text-gray-400">/100</span>
                  </div>
                  <div
                    className={`text-lg mt-2 ${BAND_COLORS[data.fresh.band].text}`}
                  >
                    {BAND_COLORS[data.fresh.band].label}
                  </div>
                  <ul className="mt-4 space-y-1 text-base text-gray-500">
                    {data.fresh.bullets.map((b, i) => (
                      <li key={i}>• {b}</li>
                    ))}
                  </ul>
                </div>

                {/* Fuel Card */}
                <div className="p-6 bg-white border border-gray-200">
                  <div className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-3">
                    FUEL WEEKLY RISK SCORE
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold text-gray-900">
                      {data.fuel.score}
                    </span>
                    <span className="text-2xl text-gray-400">/100</span>
                  </div>
                  <div
                    className={`text-lg mt-2 ${BAND_COLORS[data.fuel.band].text}`}
                  >
                    {BAND_COLORS[data.fuel.band].label}
                  </div>
                  <ul className="mt-4 space-y-1 text-base text-gray-500">
                    {data.fuel.bullets.map((b, i) => (
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
                  Policy: {data.recommendedPlan.policy}
                </p>
                <div className="mt-4 grid grid-cols-2 gap-6">
                  <div>
                    <div className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">FRESH WEEKLY RISK SCORE</div>
                    <div className="text-lg font-semibold">
                      <span className="text-gray-700">{data.recommendedPlan.fresh.baseline}</span>
                      <span className="text-gray-400 mx-2">→</span>
                      <span className="text-green-600">{data.recommendedPlan.fresh.scenario}</span>
                    </div>
                    <div className="text-base text-gray-500 mt-1">
                      ~{data.recommendedPlan.fresh.hoursAvoided}h shelf-gap
                      avoided
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">FUEL WEEKLY RISK SCORE</div>
                    <div className="text-lg font-semibold">
                      <span className="text-gray-700">{data.recommendedPlan.fuel.baseline}</span>
                      <span className="text-gray-400 mx-2">→</span>
                      <span className="text-green-600">{data.recommendedPlan.fuel.scenario}</span>
                    </div>
                    <div className="text-base text-gray-500 mt-1">
                      ~{data.recommendedPlan.fuel.trailersAvoided} trailers
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
                  {data.days.map((day, i) => (
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
                      <div className="mt-3 text-sm text-gray-600">
                        <div className="text-xs text-gray-500 mb-1">Daily Risk Scores</div>
                        <div>Fresh: {day.fresh} · Fuel: {day.fuel}</div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Sailing Table (inline expansion) */}
                <div className="mt-8">
                  <div className="flex justify-between items-center mb-4">
                    <div className="text-base font-medium text-gray-900">
                      Sailings | {data.days[selectedDay]?.name || ""}{" "}
                      {data.days[selectedDay]?.date || ""}
                    </div>
                    <button
                      onClick={exportSailingsCSV}
                      className="text-sm text-gray-500 hover:text-gray-700"
                    >
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
                            WOTDI
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            BSEF
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Gust
                          </th>
                          <th className="text-left py-3 font-medium text-gray-500 uppercase tracking-wide">
                            Tide
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
                        {(data.sailingsByDay[selectedDay] || []).length > 0 ? (
                          (data.sailingsByDay[selectedDay] || []).map((s, i) => (
                            <tr
                              key={i}
                              className="border-b border-gray-100"
                            >
                              <td className="py-3 text-gray-600">{s.time}</td>
                              <td className="py-3 font-medium text-gray-900">
                                {s.id}
                              </td>
                              <td className="py-3 text-gray-600">{s.route}</td>
                              <td className="py-3 text-gray-600">
                                {s.wotdi ?? "-"}
                              </td>
                              <td className="py-3 text-gray-600">
                                {s.bsef ?? "-"}
                              </td>
                              <td className="py-3 text-gray-600">
                                {s.gust ?? "-"}
                              </td>
                              <td className="py-3 text-gray-600">
                                {s.tide ?? "-"}
                              </td>
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
                          ))
                        ) : (
                          <tr>
                            <td colSpan={10} className="py-8 text-center text-gray-500">
                              No sailings scheduled for this day
                            </td>
                          </tr>
                        )}
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
                  Scenario A: {data.scenarios.A.name}
                </button>
                <button
                  onClick={() => setSelectedScenario("B")}
                  className={`px-6 py-3 text-lg font-medium border-b-2 transition-all ${
                    selectedScenario === "B"
                      ? "border-gray-900 text-gray-900"
                      : "border-transparent text-gray-500 hover:text-gray-700"
                  }`}
                >
                  Scenario B: {data.scenarios.B.name}
                </button>
              </div>

              {/* Impact Summary */}
              <div className="bg-amber-50 p-6 border-2 border-amber-200">
                <div className="text-xl font-semibold text-gray-900 mb-4">
                  Impact Summary
                </div>
                <p className="text-lg font-medium text-gray-700 mb-6">
                  Policy: {data.scenarios[selectedScenario].description}
                </p>
                <div className="grid grid-cols-2 gap-6">
                  <div className="p-4 bg-white border border-gray-200">
                    <div className="text-base font-medium text-gray-900 mb-2">
                      Fresh Risk
                    </div>
                    <div className="text-5xl font-bold text-gray-900">
                      {data.scenarios[selectedScenario].fresh.baseline} →{" "}
                      {data.scenarios[selectedScenario].fresh.scenario}
                    </div>
                    <div className="text-lg mt-2">
                      <span className={getRiskInfo(data.scenarios[selectedScenario].fresh.baseline).colorClass}>
                        {getRiskInfo(data.scenarios[selectedScenario].fresh.baseline).label}
                      </span> →{" "}
                      <span className={getRiskInfo(data.scenarios[selectedScenario].fresh.scenario).colorClass}>
                        {getRiskInfo(data.scenarios[selectedScenario].fresh.scenario).label}
                      </span>
                    </div>
                    <div className="text-base text-gray-500 mt-2">
                      ~{data.scenarios[selectedScenario].fresh.hoursAvoided}h
                      shelf-gap avoided
                    </div>
                  </div>
                  <div className="p-4 bg-white border border-gray-200">
                    <div className="text-base font-medium text-gray-900 mb-2">
                      Fuel Risk
                    </div>
                    <div className="text-5xl font-bold text-gray-900">
                      {data.scenarios[selectedScenario].fuel.baseline} →{" "}
                      {data.scenarios[selectedScenario].fuel.scenario}
                    </div>
                    <div className="text-lg mt-2">
                      <span className={getRiskInfo(data.scenarios[selectedScenario].fuel.baseline).colorClass}>
                        {getRiskInfo(data.scenarios[selectedScenario].fuel.baseline).label}
                      </span> →{" "}
                      <span className={getRiskInfo(data.scenarios[selectedScenario].fuel.scenario).colorClass}>
                        {getRiskInfo(data.scenarios[selectedScenario].fuel.scenario).label}
                      </span>
                    </div>
                    <div className="text-base text-gray-500 mt-2">
                      ~{data.scenarios[selectedScenario].fuel.trailersAvoided}{" "}
                      trailers saved
                    </div>
                  </div>
                </div>
              </div>

              {/* Sailing Actions Table */}
              <div className="bg-white p-6 border border-gray-200">
                <div className="flex justify-between items-center mb-4">
                  <div className="text-xl font-semibold text-gray-900">
                    Per-Sailing Suggested Actions
                  </div>
                  <button
                    onClick={exportScenarioCSV}
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    Export CSV
                  </button>
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
