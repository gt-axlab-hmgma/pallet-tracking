/**
 * Leading and trailing throttle. The first call runs immediately, later calls inside the window
 * are collapsed into one trailing call with the most recent arguments. `flush` runs a pending
 * call now (used on pointer-up so the final position is always sent).
 */
export function createThrottle<A extends unknown[]>(fn: (...args: A) => void, ms: number) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: A | null = null;

  const fire = () => {
    timer = null;
    if (pending) {
      const args = pending;
      pending = null;
      fn(...args);
      timer = setTimeout(fire, ms);
    }
  };

  return {
    call(...args: A) {
      if (timer === null) {
        fn(...args);
        timer = setTimeout(fire, ms);
      } else {
        pending = args;
      }
    },
    flush() {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      if (pending) {
        const args = pending;
        pending = null;
        fn(...args);
      }
    },
    cancel() {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}
