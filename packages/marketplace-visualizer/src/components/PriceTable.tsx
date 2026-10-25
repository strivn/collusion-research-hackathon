import { clsx } from "clsx";
import React, { useMemo } from "react";

import { AnalyticsData, Business, SellerRound } from "../types";
import { buildPriceLookup, priceKey, PriceLookup, sellerName } from "../utils/prices";

// One price cell: the price, its change from the previous round, and units sold
const PriceCell: React.FC<{ row?: SellerRound; previous?: SellerRound }> = ({ row, previous }) => {
  if (!row) return <span className="text-gray-300">–</span>;
  const change = previous ? row.price - previous.price : 0;
  return (
    <div className="leading-tight">
      <div className="font-semibold tabular-nums text-gray-800">${row.price.toFixed(2)}</div>
      {Math.abs(change) >= 0.005 && (
        <div
          className={clsx(
            "text-[10px] tabular-nums",
            change > 0 ? "text-red-600" : "text-emerald-600",
          )}
        >
          {change > 0 ? "▲" : "▼"} {Math.abs(change).toFixed(2)}
        </div>
      )}
      <div className={clsx("text-[10px]", row.units_sold > 0 ? "text-brand-700" : "text-gray-400")}>
        {row.units_sold} sold
      </div>
    </div>
  );
};

interface RoundPriceGridProps {
  round: number;
  sellers: string[];
  items: string[];
  lookup: PriceLookup;
  businesses: Business[];
  highlightSeller?: string;
}

// Prices of one round: one row per seller, one column per item
export const RoundPriceGrid: React.FC<RoundPriceGridProps> = ({
  round,
  sellers,
  items,
  lookup,
  businesses,
  highlightSeller,
}) => (
  <div className="overflow-x-auto">
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="py-1 pr-3 font-medium">Seller</th>
          {items.map((item) => (
            <th key={item} className="px-2 py-1 font-medium">
              {item}
            </th>
          ))}
          <th className="px-2 py-1 font-medium">Revenue</th>
        </tr>
      </thead>
      <tbody>
        {sellers.map((seller) => {
          const revenue = items.reduce(
            (sum, item) => sum + (lookup.get(priceKey(round, seller, item))?.revenue ?? 0),
            0,
          );
          return (
            <tr
              key={seller}
              className={clsx(
                "border-t border-gray-100 align-top",
                seller === highlightSeller && "bg-brand-50",
              )}
            >
              <td className="py-1.5 pr-3 font-medium text-gray-700">
                {sellerName(businesses, seller)}
                <div className="text-[10px] font-normal text-gray-400">{seller}</div>
              </td>
              {items.map((item) => (
                <td key={item} className="px-2 py-1.5">
                  <PriceCell
                    row={lookup.get(priceKey(round, seller, item))}
                    previous={lookup.get(priceKey(round - 1, seller, item))}
                  />
                </td>
              ))}
              <td className="px-2 py-1.5 font-semibold tabular-nums text-gray-800">
                ${revenue.toFixed(2)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

interface PriceTableProps {
  sellerRounds: SellerRound[];
  rounds: number[];
  businesses: Business[];
  roundAnalytics: Record<string, AnalyticsData>;
  selectedRound: number | null;
  selectedBusiness: Business | null;
}

// Every seller's price per item over all rounds, with each round's split of value
const PriceTable: React.FC<PriceTableProps> = ({
  sellerRounds,
  rounds,
  businesses,
  roundAnalytics,
  selectedRound,
  selectedBusiness,
}) => {
  const lookup = useMemo(() => buildPriceLookup(sellerRounds), [sellerRounds]);
  const sellers = useMemo(
    () => [...new Set(sellerRounds.map((r) => r.seller))].sort(),
    [sellerRounds],
  );
  const items = useMemo(() => [...new Set(sellerRounds.map((r) => r.item))].sort(), [sellerRounds]);

  const roundHeader = (round: number) => (
    <th
      key={round}
      className={clsx(
        "px-2 py-1 font-medium",
        round === selectedRound ? "bg-brand-100 text-brand-800" : "text-gray-500",
      )}
    >
      Round {round}
    </th>
  );
  const summaryRows: Array<[string, (a: AnalyticsData) => string]> = [
    ["Customer share", (a) => `$${a.marketplace_summary.total_utility.toFixed(2)}`],
    ["Seller share", (a) => `$${a.marketplace_summary.seller_share.toFixed(2)}`],
    ["Lost", (a) => `$${a.marketplace_summary.lost.toFixed(2)}`],
    ["Sales", (a) => String(a.marketplace_summary.sales)],
  ];

  return (
    <div className="space-y-6">
      {/* Split of value per round */}
      <section>
        <h3 className="mb-2 text-sm font-semibold text-gray-800">Value split per round</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left">
              <th className="py-1 pr-3 font-medium text-gray-500"></th>
              {rounds.map(roundHeader)}
            </tr>
          </thead>
          <tbody>
            {summaryRows.map(([label, format]) => (
              <tr key={label} className="border-t border-gray-100">
                <td className="py-1.5 pr-3 font-medium text-gray-700">{label}</td>
                {rounds.map((round) => {
                  const a = roundAnalytics[String(round)];
                  return (
                    <td
                      key={round}
                      className={clsx(
                        "px-2 py-1.5 tabular-nums text-gray-800",
                        round === selectedRound && "bg-brand-50",
                      )}
                    >
                      {a ? format(a) : "–"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* One table per item: sellers down, rounds across */}
      {items.map((item) => (
        <section key={item}>
          <h3 className="mb-2 text-sm font-semibold text-gray-800">{item}</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left">
                <th className="py-1 pr-3 font-medium text-gray-500">Seller</th>
                {rounds.map(roundHeader)}
              </tr>
            </thead>
            <tbody>
              {sellers
                .filter((seller) => lookup.has(priceKey(rounds[0], seller, item)))
                .map((seller) => (
                  <tr
                    key={seller}
                    className={clsx(
                      "border-t border-gray-100 align-top",
                      seller === selectedBusiness?.id && "bg-brand-50",
                    )}
                  >
                    <td className="py-1.5 pr-3 font-medium text-gray-700">
                      {sellerName(businesses, seller)}
                      <div className="text-[10px] font-normal text-gray-400">{seller}</div>
                    </td>
                    {rounds.map((round) => (
                      <td
                        key={round}
                        className={clsx("px-2 py-1.5", round === selectedRound && "bg-brand-50")}
                      >
                        <PriceCell
                          row={lookup.get(priceKey(round, seller, item))}
                          previous={lookup.get(priceKey(round - 1, seller, item))}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  );
};

export default PriceTable;
