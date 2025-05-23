[varunneal.github.io](https://varunneal.github.io)

Built using Quartz v4

<details>
<summary>Hints:</summary>

- `nvm use 22`
- build locally `npx quartz build --serve`
- push to git `npx quartz sync`

In order to make transparent edge-detected images I'm using ImageMagick with commands like

```bash
 convert input.jpg -alpha set -background none -canny 0x0.5+8%+25% -negate -transparent white output.png
```
</details>


