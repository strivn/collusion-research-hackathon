import { ArrowRight, Users } from "lucide-react";
import React, { useMemo } from "react";

import { Business, SellerMessage, SellerRound } from "../types";
import { getBusinessAvatar } from "../utils/avatars";
import { buildPriceLookup, sellerName } from "../utils/prices";
import { RoundPriceGrid } from "./PriceTable";

interface SellerMessagesProps {
  sellerMessages: SellerMessage[];
  sellerRounds: SellerRound[];
  rounds: number[];
  businesses: Business[];
  selectedRound: number | null;
  selectedBusiness: Business | null;
}

// Per round: what sellers wrote to each other, then the prices they set and what sold
const SellerMessages: React.FC<SellerMessagesProps> = ({
  sellerMessages,
  sellerRounds,
  rounds,
  businesses,
  selectedRound,
  selectedBusiness,
}) => {
  const lookup = useMemo(() => buildPriceLookup(sellerRounds), [sellerRounds]);
  const sellers = useMemo(
    () => [...new Set(sellerRounds.map((r) => r.seller))].sort(),
    [sellerRounds],
  );
  const items = useMemo(() => [...new Set(sellerRounds.map((r) => r.item))].sort(), [sellerRounds]);

  const shownRounds = selectedRound === null ? rounds : [selectedRound];
  // A selected business narrows to the messages it sent or received
  const involves = (m: SellerMessage) =>
    !selectedBusiness ||
    m.sender === selectedBusiness.id ||
    m.recipients.includes(selectedBusiness.id);

  return (
    <div className="space-y-6">
      {shownRounds.map((round) => {
        // Turn order; messages within a turn keep log order
        const roundMessages = sellerMessages
          .filter((m) => m.round === round && involves(m))
          .sort((a, b) => a.turn - b.turn);
        return (
          <section key={round} className="rounded-xl border border-brand-100 p-4">
            <h3 className="mb-3 text-sm font-bold text-gray-800">Round {round}</h3>

            {/* Messages */}
            {roundMessages.length === 0 ? (
              <p className="mb-3 text-xs italic text-gray-400">No seller messages this round</p>
            ) : (
              <div className="mb-4 space-y-2">
                {roundMessages.map((m, i) => {
                  const sender = sellerName(businesses, m.sender);
                  return (
                    <div key={i} className="rounded-lg bg-gray-50 p-3">
                      <div className="mb-1.5 flex flex-wrap items-center gap-1.5 text-xs">
                        <span className="rounded-full bg-brand-100 px-2 py-0.5 font-semibold text-brand-700">
                          Turn {m.turn}
                        </span>
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-gradient-to-br from-gray-300 to-gray-400 text-[10px] font-bold text-white">
                          {getBusinessAvatar(sender)}
                        </span>
                        <span className="font-semibold text-gray-800">{sender}</span>
                        <ArrowRight className="h-3 w-3 text-brand-400" />
                        <Users className="h-3 w-3 text-gray-400" />
                        <span className="text-gray-600">
                          {m.recipients.map((r) => sellerName(businesses, r)).join(", ")}
                        </span>
                      </div>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-gray-700">
                        {m.text}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Prices set this round */}
            {sellers.length > 0 && (
              <>
                <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Prices and sales in round {round}
                </h4>
                <RoundPriceGrid
                  round={round}
                  sellers={sellers}
                  items={items}
                  lookup={lookup}
                  businesses={businesses}
                  highlightSeller={selectedBusiness?.id}
                />
              </>
            )}
          </section>
        );
      })}
    </div>
  );
};

export default SellerMessages;
