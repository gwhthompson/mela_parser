# Mela File Format

This page covers everything you need to know to make use of the file formats Mela uses to export and import recipes.

---

## Recipes.melarecipes (ZIP)

Mela uses a `melarecipes` file when exporting multiple recipes into a single file. It's basically just a zipped folder that contains a `melarecipe` (see below) file for every recipe. Renaming the extension to `.zip` will let you easily open it in the Finder, for example.

## Recipe.melarecipe (JSON)

This is the file format Mela uses to export single recipes. It's just a `JSON` file which you can open in any text editor.

| Field | Type | Description |
|-------|------|-------------|
| id | String | Identifier. For recipes imported from the web, Mela uses the URL (without schema) as identifier, otherwise just a UUID. If you're creating a melarecipe file for importing, make sure this is not empty. |
| title | String | Title of the recipe. |
| text | String | Short description of the recipe which is displayed after the title and info in Mela.<br>Supported Markdown: Links. |
| images | [String] | Array of base64 encoded images. |
| categories | [String] | Array of category names.<br>Please note that Mela currently does not allow `,` in a category name. |
| yield | String | Yield or servings. |
| prepTime | String | Preparation time. |
| cookTime | String | Cook time. |
| totalTime | String | Total time it takes to prepare and cook the dish.<br>This does not have to be the sum of prepTime and cookTime (but mostly is). |
| ingredients | String | Ingredients, separated by `\n`.<br>Supported Markdown: Links and `#` for group titles. |
| instructions | String | Instructions, separated by `\n`.<br>Supported Markdown: `#` `*` `**` and links. |
| notes | String | The notes that are displayed right after the instructions in Mela.<br>Supported Markdown: `#` `*` `**` and links. |
| nutrition | String | Nutrition information.<br>Supported Markdown: `#` `*` `**` and links. |
| link | String | This might be a bit misleading but this field does not have to contain an URL (it can). It's basically just the source of the recipe and will accept any string. |

In addition to these, the `JSON` file also contains the following fields which are ignored when importing.

| Field | Type | Description |
|-------|------|-------------|
| favorite | Bool | true or false |
| wantToCook | Bool | true or false |
| date | Double | Seconds since 00:00:00 UTC on 1 January 2001 |
