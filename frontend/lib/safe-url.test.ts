import { describe, expect, it } from "vitest";

import { safeHref } from "./safe-url";

describe("safeHref", () => {
  it("keeps http and https links", () => {
    expect(safeHref("https://fastapi.tiangolo.com/tutorial/")).toBe(
      "https://fastapi.tiangolo.com/tutorial/",
    );
  });

  it.each(["javascript:alert(1)", "data:text/html,<b>x</b>", "not a url", ""])(
    "refuses %j",
    (url) => {
      expect(safeHref(url)).toBeNull();
    },
  );
});
