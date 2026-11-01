# Groups without labels (method and settings fixed in notes/results.md): Gaussian mixture by BIC on 5 PCA
# dimensions of the page-hand units, merged to modes by ridgeline ratio, a group kept if its mean Jaccard over 1,000
# sheet resamples is at least 0.75.
# Input: outputs/unsupervised_classification/units_<stem>.csv from scripts/page_heaps_units.py.
# Output: outputs/unsupervised_classification/summary_<stem>.txt and groups_<stem>.csv.
# Usage: Rscript scripts/page_heaps.R [stem]   (default dinov3_unet_all)
suppressMessages({library(mclust); library(fpc); library(parallel)})

file_arg <- sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))
root <- normalizePath(file.path(dirname(file_arg), ".."))
pins <- do.call(rbind, strsplit(readLines(file.path(root, "requirements-r.txt")), "=="))
stopifnot(as.character(getRversion()) == pins[pins[, 1] == "R", 2],
          as.character(packageVersion("mclust")) == pins[pins[, 1] == "mclust", 2],
          as.character(packageVersion("fpc")) == pins[pins[, 1] == "fpc", 2])

args <- commandArgs(trailingOnly = TRUE)
stem <- if (length(args)) args[1] else "dinov3_unet_all"
out <- file.path(root, "outputs/unsupervised_classification")
units <- read.csv(file.path(out, paste0("units_", stem, ".csv")), colClasses = c(page = "character",
                  scribe = "character", language = "character", sheet = "character"))
emb <- as.matrix(units[, grep("^e[0-9]+$", names(units))])
DRAWS <- 1000

heaps <- function(x) {
  z <- prcomp(x, center = TRUE, scale. = FALSE, rank. = 5)$x
  bic <- mclustBIC(z, G = 1:8, verbose = FALSE)
  s <- summary(bic, z)
  if (s$G == 1) return(list(groups = rep(1L, nrow(x)), components = rep(1L, nrow(x)), G = 1, model = s$modelName, ratios = NULL))
  m <- mergenormals(z, s, method = "ridge.ratio", cutoff = 0.2)
  list(groups = m$clustering, components = s$classification, G = s$G, model = s$modelName, ratios = m$new.decisionmatrix)
}

fit <- heaps(emb)
found <- sort(unique(fit$groups))
sheets <- unique(units$sheet)
set.seed(1)
draws <- lapply(seq_len(DRAWS), function(b) sample(sheets, length(sheets), replace = TRUE))
one_draw <- function(drawn) {
  rows <- unlist(lapply(drawn, function(s) which(units$sheet == s)))
  f <- tryCatch(heaps(emb[rows, , drop = FALSE]), error = function(e) NULL)
  if (is.null(f)) return(NULL)
  keep <- !duplicated(rows)
  u <- rows[keep]
  g <- f$groups[keep]
  jac <- sapply(found, function(j) {
    a <- u[fit$groups[u] == j]
    if (!length(a)) return(NA)
    max(sapply(unique(g), function(h) { b <- u[g == h]; length(intersect(a, b)) / length(union(a, b)) }))
  })
  list(jaccard = jac, groups = length(unique(f$groups)))
}
results <- mclapply(draws, one_draw, mc.cores = max(1, detectCores() - 2))
ok <- Filter(Negate(is.null), results)
jaccard <- colMeans(do.call(rbind, lapply(ok, `[[`, "jaccard")), na.rm = TRUE)
stable <- found[jaccard >= 0.75]
answer <- if (length(found) < 2 || length(stable) < 2) {
  "no separate heaps at this resolution (means less than about 4.3 page SDs apart)"
} else sprintf("%d separate heaps", length(stable))
counts <- table(sapply(ok, `[[`, "groups"))

lines <- c(
  sprintf("%s: %d page-hand units, %d sheets, %d letters", stem, nrow(units), length(sheets), sum(units$letters)),
  sprintf("BIC: %d components (%s); after ridgeline-ratio merge (cutoff 0.2): %d groups, sizes %s",
          fit$G, fit$model, length(found), paste(table(fit$groups), collapse = ", ")),
  sprintf("mean sheet-bootstrap Jaccard per group: %s", paste(sprintf("%.2f", jaccard), collapse = ", ")),
  sprintf("ANSWER: %s", answer),
  sprintf("draws: %d of %d fitted; groups per draw: %s", length(ok), DRAWS,
          paste(sprintf("%s: %.0f%%", names(counts), 100 * counts / length(ok)), collapse = ", ")))
if (!is.null(fit$ratios)) lines <- c(lines, "ridgeline ratios between merged groups:",
                                      capture.output(print(round(fit$ratios, 3))))
ab <- units$language %in% c("A", "B")
lines <- c(lines, sprintf("description only, adjusted Rand index: Davis scribe %.3f; Currier A/B %.3f",
                          adjustedRandIndex(fit$groups, units$scribe),
                          adjustedRandIndex(fit$groups[ab], units$language[ab])))
lines <- c(lines, "before the merge, pages per BIC component by Davis scribe:",
           capture.output(print(table(component = fit$components, scribe = units$scribe))),
           sprintf("adjusted Rand index of the components: Davis scribe %.3f; Currier A/B %.3f",
                   adjustedRandIndex(fit$components, units$scribe),
                   adjustedRandIndex(fit$components[ab], units$language[ab])))
writeLines(lines, file.path(out, paste0("summary_", stem, ".txt")))
write.csv(data.frame(units[, c("page", "scribe", "language", "sheet", "letters")], group = fit$groups),
          file.path(out, paste0("groups_", stem, ".csv")), row.names = FALSE)
cat(lines, sep = "\n")
