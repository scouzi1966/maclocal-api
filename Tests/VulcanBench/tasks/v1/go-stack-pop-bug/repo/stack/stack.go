// Package stack implements a simple last-in-first-out (LIFO) stack of ints.
package stack

// Stack is a LIFO stack of ints backed by a slice.
type Stack struct {
	items []int
}

// New returns an empty Stack.
func New() *Stack {
	return &Stack{}
}

// Push adds v to the top of the stack.
func (s *Stack) Push(v int) {
	s.items = append(s.items, v)
}

// Len reports how many items are currently on the stack.
func (s *Stack) Len() int {
	return len(s.items)
}
