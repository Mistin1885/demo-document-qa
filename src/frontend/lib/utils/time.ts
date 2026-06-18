/**
 * Relative time formatter — no external dependencies.
 *
 * Rules:
 *   < 60 s     → "just now"
 *   < 60 min   → "X mins ago"
 *   < 24 h     → "X hours ago"
 *   < 7 days   → "X days ago"
 *   >= 7 days  → localised short date e.g. "Mar 5"
 */

const rtf = new Intl.RelativeTimeFormat("en", { numeric: "always" });

/** Short month names for the fallback date display. */
const SHORT_MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
] as const;

/**
 * Format an ISO-8601 timestamp (or Date) as a human-readable relative string.
 * `now` defaults to the current time; it is exposed as a parameter to make
 * the function deterministically testable.
 */
export function formatRelativeTime(
  isoOrDate: string | Date,
  now: Date = new Date()
): string {
  const date = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1_000);

  if (diffSec < 60) {
    return "just now";
  }

  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) {
    return rtf.format(-diffMin, "minute").replace("minutes", "mins").replace("minute", "min");
  }

  const diffHour = Math.floor(diffMin / 60);
  if (diffHour < 24) {
    return rtf.format(-diffHour, "hour");
  }

  const diffDay = Math.floor(diffHour / 24);
  if (diffDay < 7) {
    return rtf.format(-diffDay, "day");
  }

  // >= 7 days: show "Mon D" (no year if same year, add year if different)
  const month = SHORT_MONTHS[date.getMonth()];
  const day = date.getDate();
  if (date.getFullYear() !== now.getFullYear()) {
    return `${month} ${day}, ${date.getFullYear()}`;
  }
  return `${month} ${day}`;
}
