/**
 * A tiny event emitter.
 *
 * NOTE: `off` and `once` are not implemented correctly — see issue.md. `on`,
 * `emit`, and `listenerCount` work.
 */

export type Listener = (...args: unknown[]) => void;

export class Emitter {
  private listeners = new Map<string, Listener[]>();

  /** Subscribe `fn` to `event`. */
  on(event: string, fn: Listener): void {
    const fns = this.listeners.get(event) ?? [];
    fns.push(fn);
    this.listeners.set(event, fns);
  }

  /** Unsubscribe `fn` from `event`. */
  off(event: string, fn: Listener): void {
    // TODO: actually remove the listener.
    void event;
    void fn;
  }

  /** Subscribe `fn` to `event` for a single emit, then auto-remove it. */
  once(event: string, fn: Listener): void {
    // TODO: remove after the first emit.
    this.on(event, fn);
  }

  /** Invoke every listener subscribed to `event`, in subscription order. */
  emit(event: string, ...args: unknown[]): void {
    const fns = this.listeners.get(event);
    if (!fns) return;
    for (const fn of fns) {
      fn(...args);
    }
  }

  /** Number of listeners currently subscribed to `event`. */
  listenerCount(event: string): number {
    return (this.listeners.get(event) ?? []).length;
  }
}
