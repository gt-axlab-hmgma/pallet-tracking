import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createThrottle } from "./throttle";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("createThrottle", () => {
  it("runs the first call at once and collapses the rest into one trailing call", () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    t.call(1);
    t.call(2);
    t.call(3);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenLastCalledWith(1);
    vi.advanceTimersByTime(50);
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith(3);
  });

  it("allows a new leading call once the window has passed", () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    t.call("a");
    vi.advanceTimersByTime(200);
    t.call("b");
    expect(fn.mock.calls).toEqual([["a"], ["b"]]);
  });

  it("flush sends the pending value immediately, once", () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    t.call(1);
    t.call(2);
    t.flush();
    expect(fn.mock.calls).toEqual([[1], [2]]);
    vi.advanceTimersByTime(500);
    expect(fn).toHaveBeenCalledTimes(2);
    t.flush();
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("cancel drops the pending call", () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    t.call(1);
    t.call(2);
    t.cancel();
    vi.advanceTimersByTime(500);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
