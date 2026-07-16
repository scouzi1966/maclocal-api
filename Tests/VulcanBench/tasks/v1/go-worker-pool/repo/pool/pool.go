// Package pool runs a function over many inputs using a bounded number of
// worker goroutines, returning the results in input order.
package pool

import "sync"

// Map applies fn to every element of inputs using up to `workers` goroutines
// and returns the results in the same order as inputs. A workers value below 1
// is treated as 1.
//
// NOTE: this implementation has a bug — see issue.md.
func Map(inputs []int, workers int, fn func(int) int) []int {
	if workers < 1 {
		workers = 1
	}
	results := make([]int, len(inputs))
	jobs := make(chan int)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = fn(inputs[i])
			}
		}()
	}

	for i := 0; i < len(inputs)-1; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return results
}
