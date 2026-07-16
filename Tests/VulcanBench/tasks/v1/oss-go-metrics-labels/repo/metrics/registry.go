package metrics

type Registry struct { counters map[string]int }
func New() *Registry { return &Registry{counters: map[string]int{}} }
func (r *Registry) Inc(name string) { r.counters[name]++ }
