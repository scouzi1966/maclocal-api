/**
 * Debounce a function: delay invoking `fn` until `wait` ms have elapsed since
 * the last call.
 *
 * NOTE: this is an unimplemented skeleton — see issue.md. It returns a callable
 * with a `cancel` method, but it never actually invokes `fn`.
 */

export interface DebounceOptions {
  /** Invoke on the leading edge of the wait window (default false). */
  leading?: boolean;
  /** Invoke on the trailing edge of the wait window (default true). */
  trailing?: boolean;
}

export interface Debounced<A extends unknown[]> {
  (...args: A): void;
  /** Cancel any pending trailing invocation. */
  cancel(): void;
}

export function debounce<A extends unknown[]>(
  fn: (...args: A) => void,
  wait: number,
  options: DebounceOptions = {},
): Debounced<A> {
  // TODO: implement leading/trailing debouncing and cancel().
  void fn;
  void wait;
  void options;
  const debounced = ((..._args: A): void => {}) as Debounced<A>;
  debounced.cancel = (): void => {};
  return debounced;
}
