//go:build it

package ledger_test

// transactions_planner_hints_test.go covers the adaptive fallback for the
// transactions-list SELECT introduced as a stopgap for the sparse-wallet
// timeout (prod-us-east-1-deriv, v2.4.9, ~50 s list timeout on
// SELECT … ORDER BY id DESC LIMIT 16 with JSONB @> predicates).
//
// The four properties we verify:
//
//  1. Fast path is unchanged: when the probe succeeds within the timeout
//     the result is identical to the plain (no-fallback) path.
//
//  2. Fallback triggers correctly: a tight probe timeout causes SQLSTATE 57014
//     and the retry with the GIN override succeeds, returning the right rows.
//
//  3. SET LOCAL does not leak: after Paginate returns, planner settings and
//     statement_timeout are restored to the session defaults on the same
//     connection (tested on a pinned connection so we can verify reliably).
//
//  4. GetOne and Count are unaffected: they delegate to the base repository
//     and must return the same results as the non-adaptive store.

import (
	"context"
	"fmt"
	"math/big"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/uptrace/bun"

	logging "github.com/formancehq/go-libs/v5/pkg/observe/log"
	"github.com/formancehq/go-libs/v5/pkg/query"
	"github.com/formancehq/go-libs/v5/pkg/storage/bun/paginate"
	"github.com/formancehq/go-libs/v5/pkg/types/pointer"

	ledger "github.com/formancehq/ledger/internal"
	"github.com/formancehq/ledger/internal/storage/common"
	ledgerstore "github.com/formancehq/ledger/internal/storage/ledger"
)

// ── helpers ──────────────────────────────────────────────────────────────────

// setupHintsTestData creates walletTxCount transactions from/to "wallet:main"
// and unrelatedTxCount transactions between unrelated accounts, all in a fresh
// ledger. Returns the plain (no-adaptive-config) store.
func setupHintsTestData(t *testing.T, walletTxCount, unrelatedTxCount int) *ledgerstore.Store {
	t.Helper()
	store := newLedgerStore(t)
	ctx := logging.TestingContext()

	for i := 0; i < walletTxCount; i++ {
		tx := ledger.NewTransaction().WithPostings(
			ledger.NewPosting("wallet:main", fmt.Sprintf("account%d", i), "USD", big.NewInt(100)),
		)
		require.NoError(t, commitTransactionAndUpsertAccounts(ctx, store, &tx))
	}
	for i := 0; i < unrelatedTxCount; i++ {
		tx := ledger.NewTransaction().WithPostings(
			ledger.NewPosting("world", fmt.Sprintf("other%d", i), "USD", big.NewInt(50)),
		)
		require.NoError(t, commitTransactionAndUpsertAccounts(ctx, store, &tx))
	}
	return store
}

// storeWithConfig returns a new Store sharing the ledger/bucket/db of base but
// carrying the given TransactionListConfig. This lets tests exercise different
// timeout and fallback configurations without touching the shared test factory.
func storeWithConfig(t *testing.T, base *ledgerstore.Store, cfg ledgerstore.TransactionListConfig) *ledgerstore.Store {
	t.Helper()
	return ledgerstore.New(
		base.GetDB(),
		base.GetBucket(),
		base.GetLedger(),
		ledgerstore.WithTransactionListConfig(cfg),
	)
}

// walletQuery returns a cursor-paginated query that filters by source/destination
// "wallet:main", replicating the Deriv wallet-filter pattern.
func walletQuery(pageSize uint64) common.ColumnPaginatedQuery[any] {
	// OrderDesc is an untyped iota constant; explicit cast to paginate.Order required.
	order := paginate.Order(paginate.OrderDesc)
	return common.ColumnPaginatedQuery[any]{
		InitialPaginatedQuery: common.InitialPaginatedQuery[any]{
			Column:   "id",
			Order:    &order,
			PageSize: pageSize,
			Options: common.ResourceQuery[any]{
				Builder: query.Match("account", "wallet:main"),
			},
		},
	}
}

// showSetting queries the value of a Postgres GUC on the given connection.
func showSetting(t *testing.T, ctx context.Context, conn bun.IDB, name string) string {
	t.Helper()
	var val string
	require.NoError(t, conn.QueryRowContext(ctx, "SHOW "+name).Scan(&val))
	return val
}

// ── tests ─────────────────────────────────────────────────────────────────────

// TestTransactionListAdaptive_FastPathUnchanged verifies that when the probe
// succeeds within the timeout budget (dense wallet, or generous timeout), the
// rows returned are identical to the plain store with no adaptive config.
func TestTransactionListAdaptive_FastPathUnchanged(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 8, 5)

	// Generous timeouts so the probe never fires.
	adaptive := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: true,
		FirstAttemptTimeoutMs:  60_000,
		RetryTimeoutMs:         60_000,
	})

	q := walletQuery(15) // enough to return all 8 in one page

	baseCursor, err := base.Transactions().Paginate(ctx, q)
	require.NoError(t, err)
	require.Len(t, baseCursor.Data, 8)

	adaptiveCursor, err := adaptive.Transactions().Paginate(ctx, q)
	require.NoError(t, err)
	require.Len(t, adaptiveCursor.Data, 8)

	for i := range baseCursor.Data {
		require.Equal(t, *baseCursor.Data[i].ID, *adaptiveCursor.Data[i].ID,
			"row %d: id mismatch between baseline and adaptive result", i)
	}
}

// TestTransactionListAdaptive_FallbackTriggeredByTimeout verifies the core
// behaviour: a deliberately tight probe timeout causes the fallback to fire
// and the retry (with GIN override and a generous timeout) still returns the
// correct rows.
func TestTransactionListAdaptive_FallbackTriggeredByTimeout(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 5, 10)

	// 1 ms probe: guaranteed to time out on any real SELECT.
	// 30 s retry: plenty of time to finish with the GIN override.
	adaptive := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: true,
		FirstAttemptTimeoutMs:  1,
		RetryTimeoutMs:         30_000,
	})

	cursor, err := adaptive.Transactions().Paginate(ctx, walletQuery(15))
	require.NoError(t, err, "retry should succeed even when probe times out")
	require.Len(t, cursor.Data, 5, "all 5 wallet transactions should be returned")

	// Rows must still be in descending id order.
	for i := 1; i < len(cursor.Data); i++ {
		require.Greater(t, *cursor.Data[i-1].ID, *cursor.Data[i].ID,
			"results must be in descending id order after fallback")
	}
}

// TestTransactionListAdaptive_FallbackRowsMatchBaseline confirms that the rows
// returned after a fallback are identical to those from the plain store —
// the GIN path and the index-scan path must agree on the result set.
func TestTransactionListAdaptive_FallbackRowsMatchBaseline(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 6, 4)

	adaptive := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: true,
		FirstAttemptTimeoutMs:  1, // always triggers fallback
		RetryTimeoutMs:         30_000,
	})

	q := walletQuery(15)

	baseCursor, err := base.Transactions().Paginate(ctx, q)
	require.NoError(t, err)

	adaptiveCursor, err := adaptive.Transactions().Paginate(ctx, q)
	require.NoError(t, err)

	require.Equal(t, len(baseCursor.Data), len(adaptiveCursor.Data))
	for i := range baseCursor.Data {
		require.Equal(t, *baseCursor.Data[i].ID, *adaptiveCursor.Data[i].ID,
			"row %d: id mismatch between baseline and fallback result", i)
	}
}

// TestTransactionListAdaptive_NoLeakage pins to a dedicated connection and
// verifies that after Paginate returns, both enable_indexscan and
// statement_timeout are restored to their session defaults. This proves that
// SET LOCAL is strictly scoped to the transaction opened by the adaptive
// wrapper and cannot bleed onto subsequent queries on the same pooled
// connection.
func TestTransactionListAdaptive_NoLeakage(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 3, 2)

	// Obtain the underlying *bun.DB to open a dedicated connection.
	pool, ok := base.GetDB().(*bun.DB)
	require.True(t, ok, "GetDB() must return *bun.DB in test context")

	conn, err := pool.Conn(ctx)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })

	// Create a store pinned to the single connection so we can observe
	// settings before and after Paginate on the exact same connection.
	pinnedStore := ledgerstore.New(
		conn,
		base.GetBucket(),
		base.GetLedger(),
		ledgerstore.WithTransactionListConfig(ledgerstore.TransactionListConfig{
			EnableAdaptiveFallback: true,
			FirstAttemptTimeoutMs:  1, // triggers fallback → GIN override fires
			RetryTimeoutMs:         30_000,
		}),
	)

	// Baseline: Postgres defaults.
	require.Equal(t, "on", showSetting(t, ctx, conn, "enable_indexscan"),
		"enable_indexscan should be 'on' before Paginate")
	require.Equal(t, "0", showSetting(t, ctx, conn, "statement_timeout"),
		"statement_timeout should be '0' (disabled) before Paginate")

	_, err = pinnedStore.Transactions().Paginate(ctx, walletQuery(10))
	require.NoError(t, err)

	// After the transaction commits, SET LOCAL must have been reverted.
	require.Equal(t, "on", showSetting(t, ctx, conn, "enable_indexscan"),
		"enable_indexscan must be restored after Paginate — SET LOCAL leaked")
	require.Equal(t, "0", showSetting(t, ctx, conn, "statement_timeout"),
		"statement_timeout must be restored after Paginate — SET LOCAL leaked")
}

// TestTransactionListAdaptive_RetryAlsoTimesOut verifies that when the retry
// itself times out (RetryTimeoutMs too tight), the error is propagated to the
// caller rather than silently swallowed. Exactly one retry, no loop.
func TestTransactionListAdaptive_RetryAlsoTimesOut(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 3, 2)

	// Both timeouts are 1 ms: both attempts will be cancelled.
	adaptive := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: true,
		FirstAttemptTimeoutMs:  1,
		RetryTimeoutMs:         1,
	})

	_, err := adaptive.Transactions().Paginate(ctx, walletQuery(15))
	require.Error(t, err, "retry timeout should surface an error to the caller")
}

// TestTransactionListAdaptive_DisabledFallback verifies that when
// EnableAdaptiveFallback is false the plain code path is taken and the query
// succeeds (no overhead from the probe/retry machinery).
func TestTransactionListAdaptive_DisabledFallback(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 4, 2)

	noFallback := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: false,
	})

	cursor, err := noFallback.Transactions().Paginate(ctx, walletQuery(15))
	require.NoError(t, err)
	require.Len(t, cursor.Data, 4)
}

// TestTransactionListAdaptive_PaginationCursorIntegrity verifies that the
// next-page cursor produced during a fallback-triggered Paginate can be decoded
// and used to fetch the next page correctly.
func TestTransactionListAdaptive_PaginationCursorIntegrity(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 5, 5)

	// Always trigger fallback so cursor is built on the retry path.
	adaptive := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: true,
		FirstAttemptTimeoutMs:  1,
		RetryTimeoutMs:         30_000,
	})

	// First page: 3 of 5 wallet txns.
	page1, err := adaptive.Transactions().Paginate(ctx, walletQuery(3))
	require.NoError(t, err)
	require.Len(t, page1.Data, 3)
	require.True(t, page1.HasMore)
	require.NotEmpty(t, page1.Next)

	// Decode cursor and fetch second page.
	var nextQ common.ColumnPaginatedQuery[any]
	require.NoError(t, paginate.UnmarshalCursor(page1.Next, &nextQ))

	page2, err := adaptive.Transactions().Paginate(ctx, nextQ)
	require.NoError(t, err)
	require.Len(t, page2.Data, 2)
	require.False(t, page2.HasMore)

	// All 5 ids must be distinct and globally in descending order.
	allIDs := append(page1.Data, page2.Data...)
	for i := 1; i < len(allIDs); i++ {
		require.Greater(t, *allIDs[i-1].ID, *allIDs[i].ID,
			"combined pages must be in descending id order")
	}
}

// TestTransactionListAdaptive_GetOneAndCountUnaffected confirms that GetOne
// and Count on an adaptive store return the same results as the base store.
func TestTransactionListAdaptive_GetOneAndCountUnaffected(t *testing.T) {
	t.Parallel()
	ctx := logging.TestingContext()

	base := setupHintsTestData(t, 4, 2)

	adaptive := storeWithConfig(t, base, ledgerstore.TransactionListConfig{
		EnableAdaptiveFallback: true,
		FirstAttemptTimeoutMs:  1, // would trigger fallback on Paginate
		RetryTimeoutMs:         30_000,
	})

	// Count.
	baseCount, err := base.Transactions().Count(ctx, common.ResourceQuery[any]{
		Builder: query.Match("account", "wallet:main"),
	})
	require.NoError(t, err)

	adaptiveCount, err := adaptive.Transactions().Count(ctx, common.ResourceQuery[any]{
		Builder: query.Match("account", "wallet:main"),
	})
	require.NoError(t, err)
	require.Equal(t, baseCount, adaptiveCount)

	// GetOne by id.
	q := walletQuery(1)
	cursor, err := base.Transactions().Paginate(ctx, q)
	require.NoError(t, err)
	require.NotEmpty(t, cursor.Data)

	tx, err := adaptive.Transactions().GetOne(ctx, common.ResourceQuery[any]{
		Builder: query.Match("id", pointer.For(*cursor.Data[0].ID)),
	})
	require.NoError(t, err)
	require.Equal(t, *cursor.Data[0].ID, *tx.ID)
}
