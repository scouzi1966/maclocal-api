package metrics

func (r *Registry) IncL(name string, labels map[string]string) {
    r.Inc(Key(name, labels))
}
