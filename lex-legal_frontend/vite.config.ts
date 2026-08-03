import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'

import { defineConfig } from 'vite'
import type { Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'

const externalVueTemplates = (): Plugin => {
  const templateOwners = new Map<string, Set<string>>()
  const templatePattern = /<template\s+src=(["'])([^"']+)\1\s*><\/template>/

  return {
    name: 'external-vue-templates',
    enforce: 'pre',
    transform(source, moduleId) {
      const componentPath = moduleId.split('?', 1)[0]
      if (!componentPath.endsWith('.vue')) return

      const match = source.match(templatePattern)
      if (!match) return

      const templatePath = resolve(dirname(componentPath), match[2])
      const owners = templateOwners.get(templatePath) ?? new Set<string>()
      owners.add(componentPath)
      templateOwners.set(templatePath, owners)
      this.addWatchFile(templatePath)

      const template = readFileSync(templatePath, 'utf-8')
      return {
        code: source.replace(
          match[0],
          `<template>\n${template}\n</template>`,
        ),
        map: null,
      }
    },
    handleHotUpdate(context) {
      const owners = templateOwners.get(context.file)
      if (!owners) return

      const modules = [...owners]
        .map((componentPath) => (
          context.server.moduleGraph.getModuleById(componentPath)
        ))
        .filter((module) => module !== undefined)

      for (const module of modules) {
        context.server.moduleGraph.invalidateModule(module)
      }
      return modules
    },
  }
}

export default defineConfig({
  plugins: [externalVueTemplates(), vue()],
  server: {
    host: '0.0.0.0',
    port: 5176,
  },
})
