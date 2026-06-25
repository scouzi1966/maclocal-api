# `Emitter.off` and `Emitter.once` don't work

The `Emitter` class in `src/emitter.ts` supports `on`, `emit`, and
`listenerCount`, but two methods are stubbed out and behave incorrectly:

- **`off(event, fn)` does nothing.** It should remove a previously-subscribed
  listener so it is no longer called on future emits. If the same function was
  subscribed more than once, a single `off` removes one subscription. It should
  only remove the listener passed to it, leaving the others in place.
- **`once(event, fn)` never auto-removes.** It currently behaves exactly like
  `on`, so the listener keeps firing on every emit. It should fire the listener
  on the **first** emit of `event` and then unsubscribe it, so a second emit
  does not call it and `listenerCount` drops back down.

Implement both correctly. Listeners must continue to fire in subscription order,
and a `once` listener removing itself during an `emit` must not disturb the other
listeners being notified in that same `emit`.

The class lives in `src/emitter.ts`.
