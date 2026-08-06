public enum AFMMLXClearablePackagePolicy {
    public static func packageIdentifiers(
        from discoveredModels: [AFMMLXDiscoveredModel]
    ) -> [String] {
        var seenPackagePaths = Set<String>()
        return discoveredModels.compactMap { discovered in
            let packagePath = discovered.packageDirectory.standardizedFileURL.path
            guard !packagePath.isEmpty,
                  seenPackagePaths.insert(packagePath).inserted else {
                return nil
            }
            return packagePath
        }
    }
}
