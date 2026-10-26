import { Business, SellerRound } from "../types";

// Look up a seller's row for one item in one round
export type PriceLookup = Map<string, SellerRound>;

export const priceKey = (round: number, seller: string, item: string) =>
  `${round}|${seller}|${item}`;

export const buildPriceLookup = (rows: SellerRound[]): PriceLookup =>
  new Map(rows.map((row) => [priceKey(row.round, row.seller, row.item), row]));

export const sellerName = (businesses: Business[], id: string) =>
  businesses.find((b) => b.id === id)?.name.replace("agent-", "") ?? id;
