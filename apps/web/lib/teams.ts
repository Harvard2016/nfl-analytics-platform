/** Presentation metadata only. Aliases do not alter schedule or model identifiers. */
const TEAMS: Record<string, [string, string]> = {
  ARI: ["Arizona Cardinals", "#a3314b"], ATL: ["Atlanta Falcons", "#b52b3e"],
  BAL: ["Baltimore Ravens", "#6955ad"], BUF: ["Buffalo Bills", "#4279c3"],
  CAR: ["Carolina Panthers", "#46aed4"], CHI: ["Chicago Bears", "#d17638"],
  CIN: ["Cincinnati Bengals", "#ea7d36"], CLE: ["Cleveland Browns", "#c3773b"],
  DAL: ["Dallas Cowboys", "#8fa9c6"], DEN: ["Denver Broncos", "#e78148"],
  DET: ["Detroit Lions", "#4f9fcc"], GB: ["Green Bay Packers", "#4f997e"],
  HOU: ["Houston Texans", "#527b9d"], IND: ["Indianapolis Colts", "#507fc0"],
  JAX: ["Jacksonville Jaguars", "#4caaaf"], KC: ["Kansas City Chiefs", "#d94d53"],
  LA: ["Los Angeles Rams", "#518ee3"], LAC: ["Los Angeles Chargers", "#52b5eb"],
  LV: ["Las Vegas Raiders", "#b4bac0"], MIA: ["Miami Dolphins", "#5ac6c0"],
  MIN: ["Minnesota Vikings", "#9375c3"], NE: ["New England Patriots", "#6c89b2"],
  NO: ["New Orleans Saints", "#d1bd87"], NYG: ["New York Giants", "#4d79bb"],
  NYJ: ["New York Jets", "#4aa080"], PHI: ["Philadelphia Eagles", "#4ca69e"],
  PIT: ["Pittsburgh Steelers", "#e6c55c"], SEA: ["Seattle Seahawks", "#80b85f"],
  SF: ["San Francisco 49ers", "#d36761"], TB: ["Tampa Bay Buccaneers", "#d45f52"],
  TEN: ["Tennessee Titans", "#80b4d9"], WAS: ["Washington Commanders", "#bd777b"],
};
const ALIASES: Record<string, string> = { LAR: "LA", STL: "LA", OAK: "LV", SD: "LAC", JAC: "JAX", WSH: "WAS" };
export function teamVisual(id: string) {
  const key = ALIASES[id] ?? id;
  const [name, color] = TEAMS[key] ?? [id, "#9ba89d"];
  return { name, color };
}
