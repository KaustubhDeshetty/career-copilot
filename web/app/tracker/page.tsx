"use client";

import { useEffect, useMemo, useState } from "react";
import type { TrackerData, TrackerSheet } from "@/lib/types";

type SheetKey = keyof TrackerData;

export default function TrackerPage() {
  const [data, setData] = useState<TrackerData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sheet, setSheet] = useState<SheetKey>("applications");

  useEffect(() => {
    fetch("/api/tracker", { cache: "no-store" })
      .then(async (res) => {
        const body = await res.json();
        if (!res.ok) throw new Error(body.error ?? "Could not load the tracker.");
        setData(body as TrackerData);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="stack">
      <section className="hero">
        <h1>Tracker</h1>
        <p className="muted">
          Read straight from <code>applications.xlsx</code>, the same workbook the Streamlit app writes.
        </p>
      </section>

      <div className="tabs" role="tablist">
        {(
          [
            ["applications", "Applications"],
            ["interviews", "Interview practice"],
          ] as [SheetKey, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={sheet === key}
            className={sheet === key ? "tab tab-active" : "tab"}
            onClick={() => setSheet(key)}
          >
            {label}
            {data ? ` (${data[key].rows.length})` : ""}
          </button>
        ))}
      </div>

      {error && <section className="card error-card">{error}</section>}
      {!data && !error && <p className="muted">Loading…</p>}
      {data && <SortableTable sheet={data[sheet]} />}
    </div>
  );
}

function SortableTable({ sheet }: { sheet: TrackerSheet }) {
  const [sortCol, setSortCol] = useState(0);
  const [desc, setDesc] = useState(true);

  const rows = useMemo(() => {
    const copy = [...sheet.rows];
    copy.sort((a, b) => {
      const cmp = (a[sortCol] ?? "").localeCompare(b[sortCol] ?? "", undefined, { numeric: true });
      return desc ? -cmp : cmp;
    });
    return copy;
  }, [sheet, sortCol, desc]);

  if (!sheet.rows.length) {
    return <section className="card muted">Nothing saved yet. Run the assistant and click Save.</section>;
  }

  return (
    <section className="card table-wrap">
      <table>
        <thead>
          <tr>
            {sheet.headers.map((h, i) => (
              <th key={h}>
                <button
                  className="th-btn"
                  onClick={() => {
                    if (i === sortCol) setDesc(!desc);
                    else {
                      setSortCol(i);
                      setDesc(false);
                    }
                  }}
                >
                  {h} {i === sortCol ? (desc ? "↓" : "↑") : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
