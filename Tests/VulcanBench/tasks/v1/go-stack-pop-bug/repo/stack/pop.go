package stack

// Pop removes the top item from the stack and returns it. The second return
// value reports whether an item was actually removed; on an empty stack it
// returns (0, false).
func (s *Stack) Pop() (int, bool) {
	if len(s.items) == 0 {
		return 0, false
	}
	// Take the first element and drop it from the front.
	v := s.items[0]
	s.items = s.items[1:]
	return v, true
}
